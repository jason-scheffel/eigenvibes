# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""VNN, a coVariance neural network with a fixed mean readout.

A coVariance neural network (Sihag et al., "coVariance Neural Networks",
NeurIPS 2022, arXiv:2205.15856) is a graph neural network whose graph is a
covariance matrix ``S``: every layer applies learned polynomials in ``S`` to
its input channels and then a nonlinearity. For regression, the prediction is
the plain average of the last layer's outputs over all nodes and channels
(Sihag et al., "Explainable Brain Age Prediction using coVariance Neural
Networks", NeurIPS 2023, arXiv:2305.18370, eq. (4)).

The float64 operations in `VNN._forward_with_last_layer` are ordered so that
outputs and gradients match an earlier implementation bit for bit. Rewrites
that are equal in exact arithmetic change the last bits, for example
precomputed powers of ``S``, ``einsum``, ``mean`` for the readout, a fused bias
add, or contiguous copies of the strided views.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal

import torch

# On ROCm builds of PyTorch (seen with ROCm 7.2 on an AMD RX 9070 XT), a GPU
# matrix product whose first operand has more than 2**19 rows silently returns
# wrong values, which vary between calls, for rows 2**19 and above. The
# weight-mixing product in `VNN._forward_with_last_layer` has B * N rows.
ROCM_MAX_ROWS: int = 2**19


class VNN(torch.nn.Module):
    """CoVariance neural network with a fixed mean readout.

    A sample ``x`` is one value per node, a row vector of length ``N``. Layer
    ``l = 1, ..., L`` maps the ``F_{l-1}`` channels of its input, rows ``h[g]``
    of length ``N``, to ``F_l`` channels::

        y[f] = sigma_l(sum_g sum_{k=0}^{K_l - 1} W_l[f, k, g] * (h[g] @ S^k) + b_l[f])

    ``S`` is the ``(N, N)`` covariance matrix, ``b_l[f]`` is one scalar per
    output channel shared by all nodes (the papers' equations have no bias),
    and ``sigma_l`` is layer ``l``'s activation module. The first layer's input
    is ``x`` (``F_0 = 1``). ``S`` multiplies from the right; for a symmetric
    ``S``, each ``(f, g)`` pair is the coVariance filter ``sum_k w_k S^k x`` of
    Sihag et al. (2022) with ``K_l`` coefficients. The prediction is the mean
    of the last layer's output over its ``F_L`` channels and ``N`` nodes; it
    has no learned parameters.

    Parameters
    ----------
    covariance : torch.Tensor
        The matrix ``S``, shape ``(N, N)``. The model stores a float64,
        C-contiguous copy of shape ``(1, 1, N, N)`` on the CPU, detached from
        any autograd graph, as a non-persistent buffer: it moves with
        `torch.nn.Module.to` and is not in the ``state_dict``.
    widths : Sequence[int]
        Output channels ``F_1, ..., F_L`` of the ``L`` layers.
    taps : Sequence[int]
        Filter taps ``K_1, ..., K_L``, each at least 1; layer ``l`` uses
        ``S^0, ..., S^(K_l - 1)``.
    activation : Callable[[], torch.nn.Module]
        Factory for ``sigma_l``, called once per layer with no arguments, for
        example ``torch.nn.PReLU``, ``torch.nn.Tanh``, or
        ``functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)``. Any
        module that maps a ``(B, F, N)`` tensor to a tensor of the same shape
        works; elementwise activations are typical. Each layer gets its own
        module, so a learnable activation such as ``torch.nn.PReLU`` learns one
        set of parameters per layer. The modules are converted to float64 on
        the CPU.

    Attributes
    ----------
    weights : torch.nn.ParameterList
        ``weights[l - 1]`` is ``W_l``, float64, shape ``(F_l, K_l, F_{l-1})``.
    biases : torch.nn.ParameterList
        ``biases[l - 1]`` is ``b_l``, float64, shape ``(F_l, 1)``.
    activations : torch.nn.ModuleList
        ``activations[l - 1]`` is ``sigma_l``.
    covariance : torch.Tensor
        The buffer holding ``S``, float64, shape ``(1, 1, N, N)``.

    Notes
    -----
    The model is created on the CPU; move it with ``.to(device)``. Its
    ``state_dict`` holds ``weights.0, weights.1, ...``, ``biases.0, biases.1,
    ...``, and the parameters and persistent buffers of the activations, such
    as ``activations.0.weight`` for ``torch.nn.PReLU``.

    Initialization uses the CPU random number generator. For each layer in
    order, the weight and then the bias are drawn from ``U(-c, c)`` with
    ``c = 1 / sqrt(F_{l-1} * K_l)``, and then ``activation`` is called, which
    draws nothing for the activations above. The weights and biases are drawn
    as float64 on the CPU, and the factory runs with float64 temporarily set
    as the default dtype, so all random draws, including any the factory
    makes, and the parameters the factory creates do not depend on the
    caller's default dtype. The constructor then draws
    ``N * F_L`` more values and discards them. The initial parameters and the
    generator's final state therefore match the earlier implementation's when
    it ran with float64 as the default dtype and the same activation factory.

    On a ROCm build of PyTorch, a GPU matrix product with more than
    ``ROCM_MAX_ROWS = 2**19`` rows silently returns wrong values for the rows
    past that limit. One of the model's products has ``B * N`` rows, so on an
    AMD GPU `forward` and `regional` raise `ValueError` when ``B * N`` exceeds
    ``2**19``; run the data in fixed-size batches of at most ``2**19 // N``
    samples. The CPU and CUDA builds for NVIDIA GPUs have no such limit.
    """

    covariance: torch.Tensor

    def __init__(
        self,
        covariance: torch.Tensor,
        widths: Sequence[int],
        taps: Sequence[int],
        activation: Callable[[], torch.nn.Module],
    ) -> None:
        super().__init__()
        self.weights: torch.nn.ParameterList = torch.nn.ParameterList()
        self.biases: torch.nn.ParameterList = torch.nn.ParameterList()
        self.activations: torch.nn.ModuleList = torch.nn.ModuleList()
        in_width: int = 1
        width: int
        k: int
        for width, k in zip(widths, taps, strict=True):
            bound: float = 1.0 / math.sqrt(in_width * k)
            weight: torch.Tensor = torch.empty(
                width, k, in_width, dtype=torch.float64, device="cpu"
            )
            bias: torch.Tensor = torch.empty(
                width, 1, dtype=torch.float64, device="cpu"
            )
            self.weights.append(torch.nn.Parameter(weight.uniform_(-bound, bound)))
            self.biases.append(torch.nn.Parameter(bias.uniform_(-bound, bound)))
            default_dtype: torch.dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                module: torch.nn.Module = activation()
            finally:
                torch.set_default_dtype(default_dtype)
            self.activations.append(module.to(device="cpu", dtype=torch.float64))
            in_width = width
        n: int = covariance.shape[0]
        # The earlier implementation randomly initialized a readout layer of
        # N * F_L weights and then overwrote them with 1 / D. Drawing and
        # discarding as many values leaves the CPU generator in the same state
        # as the earlier implementation did when run with float64 as the
        # default dtype, so later draws, such as minibatch shuffles, match.
        torch.empty(n * in_width, dtype=torch.float64, device="cpu").uniform_()
        self.register_buffer(
            "covariance",
            torch.empty(0, dtype=torch.float64, device="cpu"),
            persistent=False,
        )
        self.set_covariance(covariance)

    def set_covariance(self, covariance: torch.Tensor) -> None:
        """Replace the covariance matrix and keep the learned filters.

        It replaces the earlier implementation's covariance swap. The filter
        coefficients do not depend on the number of nodes, so filters trained
        with one covariance matrix can be applied with another, such as the
        covariance of a different cohort. The number of nodes may change to
        some ``N'``; inputs must then have ``N'`` columns. Afterwards the model
        computes bit for bit what a model constructed with ``covariance`` and
        loaded with the same parameters computes, including a readout over
        ``N' * F_L`` values. The parameters and the random number generators
        are not touched.

        The filters are polynomials in ``S``, so rescaling ``S`` changes what
        they compute. Normalize the new matrix the same way as the one the
        filters were trained with, for example by dividing it by its trace.

        Parameters
        ----------
        covariance : torch.Tensor
            The new matrix, shape ``(N', N')``. The model stores a float64,
            C-contiguous copy of shape ``(1, 1, N', N')``, detached from any
            autograd graph, on the device of the current one.
        """
        n: int = covariance.shape[0]
        self.covariance = (
            covariance.detach()
            .reshape(1, 1, n, n)
            .to(
                device=self.covariance.device,
                dtype=torch.float64,
                copy=True,
                memory_format=torch.contiguous_format,
            )
        )

    def _forward_with_last_layer(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the predictions and the last layer's output.

        Parameters
        ----------
        x : torch.Tensor
            Input signals, float64, shape ``(B, N)``: one row per sample, one
            column per node, nodes in the order of the covariance matrix.

        Returns
        -------
        prediction : torch.Tensor
            Predictions, float64, shape ``(B,)``.
        last_layer : torch.Tensor
            Output of the last layer after its activation, float64, shape
            ``(B, F_L, N)``. For elementwise activations its strides are
            ``(N * F_L, 1, F_L)``, so it is not contiguous when ``F_L > 1``.

        Raises
        ------
        ValueError
            If PyTorch is a ROCm build, ``x`` is on the GPU, and ``B * N``
            exceeds ``ROCM_MAX_ROWS``.
        """
        batch: int = x.shape[0]
        n: int = self.covariance.shape[-1]
        if (
            torch.version.hip is not None
            and x.device.type == "cuda"
            and batch * n > ROCM_MAX_ROWS
        ):
            raise ValueError(
                f"batch size {batch} times {n} nodes exceeds {ROCM_MAX_ROWS} "
                "(2**19), above which ROCm GPU matrix products return wrong "
                "values; run the data in fixed-size batches of at most "
                f"{ROCM_MAX_ROWS // n} samples"
            )
        h: torch.Tensor = x.reshape(batch, 1, n)
        weight: torch.Tensor
        bias: torch.Tensor
        activation: torch.nn.Module
        for weight, bias, activation in zip(
            self.weights, self.biases, self.activations, strict=True
        ):
            f: int = weight.shape[0]
            k: int = weight.shape[1]
            g: int = weight.shape[2]
            # Taps h S^0, ..., h S^(K - 1): C-contiguous, each computed from
            # the previous one.
            shifted: torch.Tensor = h.reshape(batch, 1, g, n)
            taps: list[torch.Tensor] = [shifted.reshape(batch, 1, 1, g, n).contiguous()]
            for _ in range(1, k):
                shifted = torch.matmul(shifted, self.covariance)
                taps.append(shifted.reshape(batch, 1, 1, g, n))
            # Row (b, n) holds node n's value in every tap of every input
            # channel, tap-major, matching the (F, K, G) order of the weight.
            # This is a strided view; one matrix product contracts it.
            rows: torch.Tensor = (
                torch.cat(taps, dim=2).permute(0, 4, 1, 2, 3).reshape(batch, n, k * g)
            )
            mixed: torch.Tensor = torch.matmul(rows, weight.reshape(f, k * g).T)
            h = activation(mixed.transpose(1, 2) + bias)
        d: int = n * h.shape[1]
        # A product with the constant row 1 / D rather than h.mean(), which
        # rounds differently.
        readout: torch.Tensor = (
            torch.ones((1, d), dtype=torch.float64, device=h.device) / d
        )
        prediction: torch.Tensor = (h.reshape(batch, d) @ readout.T).reshape(batch)
        return prediction, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one value per sample.

        Parameters
        ----------
        x : torch.Tensor
            Input signals, float64, shape ``(B, N)``.

        Returns
        -------
        torch.Tensor
            Predictions, float64, shape ``(B,)``.

        Raises
        ------
        ValueError
            If PyTorch is a ROCm build, ``x`` is on the GPU, and ``B * N``
            exceeds ``ROCM_MAX_ROWS = 2**19``; see the class Notes.
        """
        return self._forward_with_last_layer(x)[0]

    def regional(
        self,
        x: torch.Tensor,
        *,
        kind: Literal["output", "residual", "normalized_residual"],
    ) -> torch.Tensor:
        """Compute one value per node from the last layer's output.

        All three kinds come from the same forward pass. With ``y`` the last
        layer's output, shape ``(B, F_L, N)``, and ``y_hat`` the prediction
        (Sihag et al., NeurIPS 2023, arXiv:2305.18370):

        - ``"output"``: the regional output ``p``, ``y`` averaged over its
          channels (their eq. (5)). The prediction equals the mean of ``p``
          over nodes up to rounding.
        - ``"residual"``: the regional residual ``r = p - y_hat`` (their eq.
          (7)), with ``y_hat`` computed by the model's readout, not as the mean
          of ``p``. Positive entries mark nodes whose regional output is
          above the prediction and negative entries nodes below it; the 2023
          paper compares it between diagnostic groups to find the regions
          behind a higher predicted age.
        - ``"normalized_residual"``: ``r / ||r||``, per sample. Projected onto
          the eigenvectors ``V`` of the covariance matrix, ``|r_hat @ V|`` are
          the eigenvector alignments used in the brain-age papers.

        ``p`` is computed as ``y.contiguous().mean(dim=1)``. ``mean`` adds the
        ``F_L`` channels in an order that depends on the memory layout of its
        input. Averaging a C-contiguous copy reproduces, bit for bit, regional
        outputs computed as ``y.index_select(2, nodes).mean(dim=1)`` with
        ``nodes`` the indices ``0, ..., N - 1``, which is how they were computed
        from the earlier implementation's last layer. Averaging ``y`` directly,
        whose channel axis has stride 1, can change the last bits.

        Parameters
        ----------
        x : torch.Tensor
            Input signals, float64, shape ``(B, N)``.
        kind : {"output", "residual", "normalized_residual"}
            Which values to return.

        Returns
        -------
        torch.Tensor
            Float64, shape ``(B, N)``.

        Raises
        ------
        ValueError
            If ``kind`` is not one of the three values, or if ``kind`` is
            ``"normalized_residual"`` and a sample's residual has a norm that
            is exactly zero or not finite (infinite or NaN). Also if PyTorch is
            a ROCm build, ``x`` is on the GPU, and ``B * N`` exceeds
            ``ROCM_MAX_ROWS = 2**19``; see the class Notes.
        """
        prediction: torch.Tensor
        last_layer: torch.Tensor
        prediction, last_layer = self._forward_with_last_layer(x)
        output: torch.Tensor = last_layer.contiguous().mean(dim=1)
        if kind == "output":
            return output
        residual: torch.Tensor = output - prediction[:, None]
        if kind == "residual":
            return residual
        if kind == "normalized_residual":
            norm: torch.Tensor = residual.norm(dim=1, keepdim=True)
            if (~torch.isfinite(norm) | (norm == 0)).any():
                raise ValueError(
                    "a regional residual has a zero or non-finite norm; "
                    "it has no direction"
                )
            return residual / norm
        raise ValueError(f"unknown kind {kind!r}")
