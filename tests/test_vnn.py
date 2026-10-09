# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the coVariance neural network."""

import functools

import pytest
import torch

from eigenvibes.vnn import ROCM_MAX_ROWS, VNN


def test_linear_model_matches_matrix_powers() -> None:
    """Compare a model with identity activations to explicit matrix powers.

    With `torch.nn.Identity` activations, output channel ``f`` of layer ``l``
    is ``sum_{k, g} W_l[f, k, g] * (h[g] @ S^k) + b_l[f]``. The expected values
    use `torch.linalg.matrix_power` and ``einsum`` instead of the model's
    repeated products. The second layer has five input channels, so the check
    also covers the order in which the model reads the tap and channel axes of
    the weight.
    """
    torch.manual_seed(42)
    covariance: torch.Tensor = torch.cov(torch.randn(30, 7, dtype=torch.float64).T)
    model: VNN = VNN(
        covariance, widths=[5, 3], taps=[3, 2], activation=torch.nn.Identity
    )
    x: torch.Tensor = torch.randn(4, 7, dtype=torch.float64)
    powers: torch.Tensor = torch.stack(
        [torch.linalg.matrix_power(covariance, k) for k in range(3)]
    )
    expected: torch.Tensor = x[:, None, :]
    weight: torch.Tensor
    bias: torch.Tensor
    for weight, bias in zip(model.weights, model.biases, strict=True):
        expected = (
            torch.einsum(
                "fkg,bgm,kmn->bfn", weight, expected, powers[: weight.shape[1]]
            )
            + bias
        )
    prediction: torch.Tensor
    last_layer: torch.Tensor
    prediction, last_layer = model._forward_with_last_layer(x)
    torch.testing.assert_close(last_layer, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        prediction, expected.mean(dim=(1, 2)), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        model.regional(x, kind="output"), expected.mean(dim=1), rtol=0, atol=1e-12
    )


def test_set_covariance_matches_fresh_model() -> None:
    """Check that a covariance matrix of another size can be swapped in exactly.

    After `VNN.set_covariance`, the model must give bitwise the same outputs as
    a model constructed with the new matrix and loaded with the parameters the
    first model had before the swap. A stale matrix, a readout still sized for
    the old number of nodes, or a swap that alters the parameters fails this.
    On a GPU, the new matrix must also move to the model's device. The new
    matrix requires grad, and the stored copy must not.
    """
    torch.manual_seed(42)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old: torch.Tensor = torch.cov(torch.randn(30, 9, dtype=torch.float64).T)
    new: torch.Tensor = torch.cov(torch.randn(30, 6, dtype=torch.float64).T)
    model: VNN = VNN(old, widths=[4, 3], taps=[3, 2], activation=torch.nn.PReLU)
    model.to(device)
    before: dict[str, torch.Tensor] = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    model.set_covariance(new.requires_grad_())
    assert not model.covariance.requires_grad
    fresh: VNN = VNN(new, widths=[4, 3], taps=[3, 2], activation=torch.nn.PReLU)
    fresh.to(device)
    fresh.load_state_dict(before)
    x: torch.Tensor = torch.randn(5, 6, dtype=torch.float64, device=device)
    torch.testing.assert_close(
        model._forward_with_last_layer(x),
        fresh._forward_with_last_layer(x),
        rtol=0,
        atol=0,
    )


def test_regional_matches_gathered_channel_mean() -> None:
    """Check that the regional outputs round like the earlier computation.

    Regional outputs used to be computed outside the model by gathering the
    nodes of the last layer's output with ``index_select``, which returns a
    C-contiguous copy, and then averaging over channels; with the identity
    node order, that is ``expected`` below. ``mean`` adds the channels in an
    order that depends on the memory layout, so a last layer with stride 1
    along the channels, as the earlier implementation's had, averages
    differently without a copy. The last layer has six channels because the
    two orders give the same sums for one or two channels and, on the CPU
    this check was first written on, for up to four.
    """
    torch.manual_seed(42)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covariance: torch.Tensor = torch.cov(torch.randn(40, 68, dtype=torch.float64).T)
    model: VNN = VNN(
        covariance / covariance.trace(),
        widths=[7, 4, 6],
        taps=[4, 2, 5],
        activation=functools.partial(torch.nn.LeakyReLU, negative_slope=0.1),
    ).to(device)
    x: torch.Tensor = torch.randn(64, 68, dtype=torch.float64, device=device)
    nodes: torch.Tensor = torch.arange(68, device=device)
    last_layer: torch.Tensor = model._forward_with_last_layer(x)[1]
    expected: torch.Tensor = last_layer.index_select(2, nodes).mean(dim=1)
    torch.testing.assert_close(
        model.regional(x, kind="output"), expected, rtol=0, atol=0
    )


def test_regional_residuals() -> None:
    """Check the residual kinds against the regional output and the prediction.

    The residual must subtract the model's prediction, whose readout rounds
    differently from the mean of the regional output, and the normalized
    residual must divide each sample's residual by its own norm. A residual
    that is exactly zero, here from zero biases and a zero input, has no
    direction and must raise.
    """
    torch.manual_seed(42)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covariance: torch.Tensor = torch.cov(torch.randn(40, 20, dtype=torch.float64).T)
    model: VNN = VNN(
        covariance / covariance.trace(),
        widths=[6, 5],
        taps=[3, 2],
        activation=torch.nn.Tanh,
    ).to(device)
    x: torch.Tensor = torch.randn(8, 20, dtype=torch.float64, device=device)
    residual: torch.Tensor = model.regional(x, kind="residual")
    torch.testing.assert_close(
        residual,
        model.regional(x, kind="output") - model(x)[:, None],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.regional(x, kind="normalized_residual"),
        residual / residual.norm(dim=1, keepdim=True),
        rtol=0,
        atol=0,
    )
    bias: torch.Tensor
    with torch.no_grad():
        for bias in model.biases:
            bias.zero_()
    with pytest.raises(ValueError, match="zero"):
        model.regional(torch.zeros_like(x), kind="normalized_residual")


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs a ROCm build of PyTorch and a GPU",
)
def test_rocm_row_limit() -> None:
    """Check that the ROCm guard allows 2**19 rows and rejects one more.

    With one channel, the largest products have one row per sample, so the
    batch size is the row count.
    """
    model: VNN = VNN(
        torch.ones(1, 1, dtype=torch.float64),
        widths=[1],
        taps=[1],
        activation=torch.nn.Identity,
    ).to("cuda")
    x: torch.Tensor = torch.ones(
        ROCM_MAX_ROWS + 1, 1, dtype=torch.float64, device="cuda"
    )
    assert model(x[:ROCM_MAX_ROWS]).shape == (ROCM_MAX_ROWS,)
    with pytest.raises(ValueError, match="at most 524288 samples"):
        model(x)
