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

The forward pass works in the eigenbasis of ``S``. With
``S = V diag(lambda) V^T``, a filter ``sum_k w_k S^k`` equals
``V diag(p(lambda)) V^T`` with ``p(lambda) = sum_k w_k lambda^k``, so each layer
rotates its input into eigen coordinates, scales every coordinate by its
filter response, and rotates back. In exact arithmetic this equals applying
``S`` repeatedly, but its cost no longer grows with the number of taps. The
outputs and gradients match an earlier implementation, which applied ``S``
repeatedly, up to rounding; the initial parameters match it exactly.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal

import torch

# On ROCm builds of PyTorch (seen with ROCm 7.2 on an AMD RX 9070 XT), GPU
# matrix products with more than 2**19 rows can silently return wrong values,
# which vary between calls. Run past that limit, this model gave wrong outputs
# and gradients for most architectures tested, so `VNN.max_batch` keeps every
# product at or below 2**19 rows. The largest products have one row per sample
# and channel, B * max(F_1, ..., F_L) rows in all.
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
        The matrix ``S``, shape ``(N, N)``, symmetric. The model stores a float64,
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
    eigenvalues : torch.Tensor
        The buffer holding the eigenvalues ``lambda`` of ``S`` in ascending
        order, float64, shape ``(N,)``.
    eigenvectors : torch.Tensor
        The buffer holding the matching orthonormal eigenvectors ``V`` of
        ``S`` as columns, float64, shape ``(N, N)``.

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
    ``ROCM_MAX_ROWS = 2**19`` rows can silently return wrong values for the
    rows past that limit. The model's largest products have
    ``B * max(F_1, ..., F_L)`` rows, so on an AMD GPU `forward` and `regional`
    raise `ValueError` when ``B`` exceeds `max_batch`; run the data in
    fixed-size batches of at most `max_batch` samples. The CPU and CUDA builds
    for NVIDIA GPUs have no such limit.
    """

    covariance: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor

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
        name: str
        for name in ("covariance", "eigenvalues", "eigenvectors"):
            self.register_buffer(
                name,
                torch.empty(0, dtype=torch.float64, device="cpu"),
                persistent=False,
            )
        self.set_covariance(covariance)

    @property
    def max_batch(self) -> int:
        """The largest batch whose matrix products stay within 2**19 rows.

        It is ``ROCM_MAX_ROWS // max(F_1, ..., F_L)``. Larger batches raise on
        a ROCm GPU. Other devices accept any batch, but using this size on
        every device keeps the batch size, and so the rounding, the same.
        """
        widest: int = max(int(weight.shape[0]) for weight in self.weights)
        return ROCM_MAX_ROWS // widest

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
            The new matrix, shape ``(N', N')``, symmetric. The model stores a
            float64, C-contiguous copy of shape ``(1, 1, N', N')``, detached
            from any autograd graph, on the device of the current one, and its
            eigendecomposition, computed on the CPU so that it is the same for
            every device.

        Raises
        ------
        ValueError
            If ``covariance`` is not symmetric to within ``1e-12`` of its
            largest absolute entry. The eigendecomposition reads only the lower
            triangle, so a nonsymmetric matrix would silently be replaced by a
            different one.
        """
        n: int = covariance.shape[0]
        matrix: torch.Tensor = covariance.detach().to(device="cpu", dtype=torch.float64)
        if (matrix - matrix.T).abs().max() > 1e-12 * matrix.abs().max():
            raise ValueError("the covariance matrix must be symmetric")
        eigenvalues: torch.Tensor
        eigenvectors: torch.Tensor
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        device: torch.device = self.covariance.device
        self.eigenvalues = eigenvalues.to(device)
        self.eigenvectors = eigenvectors.contiguous().to(device)
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
            ``(B, F_L, N)``. For elementwise activations it is C-contiguous.

        Raises
        ------
        ValueError
            If PyTorch is a ROCm build, ``x`` is on the GPU, and ``B`` exceeds
            `max_batch`.
        """
        batch: int = x.shape[0]
        n: int = self.covariance.shape[-1]
        if (
            torch.version.hip is not None
            and x.device.type == "cuda"
            and batch > self.max_batch
        ):
            raise ValueError(
                f"batch size {batch} gives matrix products with more than "
                f"{ROCM_MAX_ROWS} (2**19) rows, above which ROCm GPU matrix "
                "products can return wrong values; run the data in fixed-size "
                f"batches of at most {self.max_batch} samples"
            )
        h: torch.Tensor = x.reshape(batch, 1, n)
        # A contiguous copy of V^T, so that no product has a transposed right
        # operand, the layout in which the ROCm bug was first seen.
        rotate_back: torch.Tensor = self.eigenvectors.T.contiguous()
        weight: torch.Tensor
        bias: torch.Tensor
        activation: torch.nn.Module
        for weight, bias, activation in zip(
            self.weights, self.biases, self.activations, strict=True
        ):
            # powers[k, i] = lambda_i^k, and response[f, g, i] = p(lambda_i)
            # for the filter from input channel g to output channel f.
            powers: torch.Tensor = self.eigenvalues ** torch.arange(
                weight.shape[1], dtype=torch.float64, device=x.device
            ).reshape(-1, 1)
            response: torch.Tensor = torch.einsum("fkg,ki->fgi", weight, powers)
            spectral: torch.Tensor = torch.einsum(
                "bgi,fgi->bfi", h @ self.eigenvectors, response
            )
            h = activation(spectral @ rotate_back + bias)
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
            If PyTorch is a ROCm build, ``x`` is on the GPU, and ``B`` exceeds
            `max_batch`; see the class Notes.
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
        from the earlier implementation's last layer, whose channel axis had
        stride 1. For such a layout, averaging ``y`` directly can change the
        last bits.

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
            a ROCm build, ``x`` is on the GPU, and ``B`` exceeds `max_batch`;
            see the class Notes.
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
