# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how fast VNNs train on a GPU, to decide how to run the experiments.

The script answers two questions and prints a table for each:

1. How long does one epoch of training take, one model at a time? It times
   VNNs with 1, 2, 3, and 4 layers (128 channels and 8 taps per layer) at
   batch sizes 2, 8, and 32.
2. Is it faster overall to train several models at the same time on one GPU
   than one after another? It trains 1, 2, 4, 8, and 16 copies of the 3-layer
   model at once, at each of the same batch sizes, and compares their total
   epochs per second with that of a single copy. Only the 3-layer model is
   tested this way.

Every model trains on the same synthetic data, 2000 rows of 12 features, with
Adam and mean squared error. The timing leaves out the validation pass that
real training runs after each epoch. Each model trains in its own process: one
untimed warm-up epoch, then the timed epochs, about 2000 training steps.
Models trained at the same time start their timed epochs together. The script
only prints and writes no files. It needs a GPU.

Run it with ``uv run python scripts/benchmark.py``; on the DGX, submit
``scripts/benchmark.sbatch`` instead.
"""

import multiprocessing
import multiprocessing.queues
import platform
import queue
import time
from multiprocessing.synchronize import Barrier

import numpy as np
import torch

from eigenvibes.vnn import VNN

ROWS: int = 2000
FEATURES: int = 12
# Timed epochs per batch size, about 2000 training steps each.
EPOCHS: dict[int, int] = {2: 2, 8: 8, 32: 32}
LEARNING_RATE: float = 1e-3
# Layer widths and filter taps of the models to time.
MODELS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "1 layer": ((128,), (8,)),
    "2 layers": ((128, 128), (8, 8)),
    "3 layers": ((128, 128, 128), (8, 8, 8)),
    "4 layers": ((128, 128, 128, 128), (8, 8, 8, 8)),
}
BATCH_SIZES: tuple[int, ...] = tuple(EPOCHS)
# The model trained several at a time in question 2.
SHARED_MODEL: str = "3 layers"
MODELS_AT_ONCE: tuple[int, ...] = (1, 2, 4, 8, 16)


def train(
    model_name: str,
    batch_size: int,
    barrier: Barrier,
    seconds: multiprocessing.queues.Queue[float],
) -> None:
    """Train one VNN on the GPU and report its mean seconds per timed epoch.

    Parameters
    ----------
    model_name : str
        Key of `MODELS`.
    batch_size : int
        Samples per training step.
    barrier : multiprocessing.synchronize.Barrier
        Shared by all models trained at the same time; each waits here after
        its warm-up epoch so that the timed epochs run together.
    seconds : multiprocessing.queues.Queue
        Receives the mean seconds per timed epoch.
    """
    # One thread each, so 16 processes don't fight over the CPUs.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    rng: np.random.Generator = np.random.default_rng(42)
    mixing: np.ndarray = rng.standard_normal((FEATURES, FEATURES)) / FEATURES**0.5
    x_np: np.ndarray = rng.standard_normal((ROWS, FEATURES)) @ mixing
    x: torch.Tensor = torch.tensor(x_np, dtype=torch.float64, device="cuda")
    y: torch.Tensor = x[:, 0] + x[:, 1] * x[:, 2]
    covariance: torch.Tensor = torch.cov(x.T)
    widths: tuple[int, ...]
    taps: tuple[int, ...]
    widths, taps = MODELS[model_name]
    model: VNN = VNN(covariance / covariance.trace(), widths, taps, torch.nn.PReLU)
    model.to("cuda")
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator: torch.Generator = torch.Generator().manual_seed(0)

    def epoch() -> None:
        order: torch.Tensor = torch.randperm(ROWS, generator=generator).to("cuda")
        index: torch.Tensor
        for index in torch.split(order, batch_size):
            optimizer.zero_grad()
            loss: torch.Tensor = torch.nn.functional.mse_loss(model(x[index]), y[index])
            torch.autograd.backward(loss)
            optimizer.step()
        torch.cuda.synchronize()  # GPU calls return before the work is done

    epoch()
    barrier.wait()
    start: float = time.perf_counter()
    _: int
    for _ in range(EPOCHS[batch_size]):
        epoch()
    seconds.put((time.perf_counter() - start) / EPOCHS[batch_size])


def measure(
    models_at_once: int, model_name: str, batch_size: int
) -> tuple[float, float]:
    """Train ``models_at_once`` copies of a model on the GPU at the same time.

    Parameters
    ----------
    models_at_once : int
        Number of models trained at the same time, each in its own process.
    model_name : str
        Key of `MODELS`.
    batch_size : int
        Samples per training step.

    Returns
    -------
    seconds_per_epoch : float
        Seconds per epoch of the slowest model.
    epochs_per_second : float
        Epochs per second of all models together, counted until the slowest
        finishes.
    """
    context: multiprocessing.context.SpawnContext = multiprocessing.get_context("spawn")
    barrier: Barrier = context.Barrier(models_at_once)
    seconds: multiprocessing.queues.Queue[float] = context.Queue()
    workers: list[multiprocessing.context.SpawnProcess] = [
        context.Process(
            target=train, args=(model_name, batch_size, barrier, seconds), daemon=True
        )
        for _ in range(models_at_once)
    ]
    worker: multiprocessing.context.SpawnProcess
    for worker in workers:
        worker.start()
    per_epoch: list[float] = []
    while len(per_epoch) < models_at_once:
        try:
            per_epoch.append(seconds.get(timeout=10))
        except queue.Empty:
            if any(worker.exitcode for worker in workers):  # a worker crashed
                for worker in workers:
                    worker.terminate()
                raise RuntimeError("a training process failed; see its error above")
    for worker in workers:
        worker.join()
    return max(per_epoch), models_at_once / max(per_epoch)


def main() -> None:
    """Print the hardware, then answer both questions."""
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found; this benchmark only measures GPU training.")
    print(f"torch {torch.__version__}, Python {platform.python_version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Each model: {ROWS} rows, 1 warm-up epoch, then about 2000 timed steps;"
        " 128 channels and 8 taps per layer."
    )

    print("\n1. Seconds per epoch, one model at a time")
    print(f"{'model':<10}" + "".join(f"  batch {b:>2}" for b in BATCH_SIZES))
    model_name: str
    batch_size: int
    for model_name in MODELS:
        row: str = f"{model_name:<10}"
        for batch_size in BATCH_SIZES:
            row += f"  {measure(1, model_name, batch_size)[0]:>8.3f}"
        print(row, flush=True)

    print(f"\n2. Several {SHARED_MODEL} models at the same time on one GPU")
    print("A speedup above 1x means sharing the GPU beats one model at a time.")
    for batch_size in BATCH_SIZES:
        print(f"\nbatch size {batch_size}")
        print("models at once  s/epoch (slowest)  epochs/s (all models)  speedup")
        single: float = 0.0
        count: int
        for count in MODELS_AT_ONCE:
            seconds: float
            total: float
            seconds, total = measure(count, SHARED_MODEL, batch_size)
            single = single or total
            print(
                f"{count:>14}  {seconds:>17.3f}  {total:>21.2f}  {total / single:>6.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
