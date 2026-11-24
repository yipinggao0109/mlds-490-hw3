"""FedAvg training, evaluation, and plotting."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .data import ClientData, FederatedDataset
from .model import TwoLayerNet

StateDict = dict[str, Tensor]


@dataclass(frozen=True)
class FedAvgConfig:
    rounds: int = 40
    client_fraction: float = 0.10
    local_epochs: int = 2
    learning_rate: float = 0.05
    batch_size: int = 64
    momentum: float = 0.0
    seed: int = 490
    device: str = "auto"
    noise_scale: float = 0.0
    noise_seed: int = 10_490

    def validate(self, n_clients: int) -> None:
        if self.rounds <= 0 or self.local_epochs <= 0:
            raise ValueError("rounds and local_epochs must be positive")
        if not 0 < self.client_fraction <= 0.10:
            raise ValueError("client_fraction must lie in (0, 0.10]")
        if round(self.client_fraction * n_clients) < 1:
            raise ValueError("client_fraction selects no clients")
        if self.learning_rate < 0 or self.batch_size <= 0:
            raise ValueError("learning_rate must be nonnegative and batch_size positive")
        if self.momentum < 0 or self.noise_scale < 0:
            raise ValueError("momentum and noise_scale must be nonnegative")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clone_state(state: StateDict) -> StateDict:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def aggregate_states(states: list[StateDict], sample_counts: list[int]) -> StateDict:
    """Compute sample-size-weighted FedAvg on CPU tensors."""

    if not states or len(states) != len(sample_counts):
        raise ValueError("states and sample_counts must be nonempty and aligned")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("all sample counts must be positive")
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError("all state dictionaries must have the same keys")
    total = float(sum(sample_counts))
    result: StateDict = {}
    for key in keys:
        first = states[0][key].detach().cpu()
        if first.is_floating_point():
            averaged = torch.zeros_like(first)
            for state, count in zip(states, sample_counts, strict=True):
                averaged.add_(state[key].detach().cpu(), alpha=count / total)
            result[key] = averaged
        else:
            result[key] = first.clone()
    return result


def perturb_images(
    images: NDArray[np.float32], scale: float, seed: int
) -> NDArray[np.float32]:
    """Apply the assignment's independent zero-centered Laplace mechanism."""

    if scale < 0:
        raise ValueError("noise scale must be nonnegative")
    if scale == 0:
        return images
    generator = np.random.default_rng(seed)
    noise = generator.laplace(0.0, scale, size=images.shape).astype(np.float32)
    return np.asarray(images + noise, dtype=np.float32)


def train_local_model(
    global_state: StateDict,
    client: ClientData,
    config: FedAvgConfig,
    client_id: int,
    round_index: int,
    device: torch.device,
) -> tuple[StateDict, dict[str, float]]:
    """Train a private client copy and return only model parameters and metrics."""

    model = TwoLayerNet().to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )
    images_np = perturb_images(
        client.train_images, config.noise_scale, config.noise_seed + client_id
    )
    images = torch.as_tensor(images_np, dtype=torch.float32)
    labels = torch.as_tensor(client.train_labels, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + round_index * 10_000 + client_id)
    loss_sum = 0.0
    correct = 0
    examples = 0
    for _ in range(config.local_epochs):
        permutation = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(labels), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_images = images[indices].to(device)
            batch_labels = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = len(indices)
            loss_sum += float(loss.detach().cpu()) * batch_size
            correct += int((logits.argmax(dim=1) == batch_labels).sum().detach().cpu())
            examples += batch_size
    return clone_state(model.state_dict()), {
        "local_loss": loss_sum / examples,
        "local_accuracy": correct / examples,
    }


def evaluate_arrays(
    model: nn.Module,
    images: NDArray[np.float32],
    labels: NDArray[np.int64],
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch_images = torch.as_tensor(
                images[start : start + batch_size], dtype=torch.float32, device=device
            )
            batch_labels = torch.as_tensor(
                labels[start : start + batch_size], dtype=torch.long, device=device
            )
            logits = model(batch_images)
            loss_sum += float(
                nn.functional.cross_entropy(logits, batch_labels, reduction="sum")
                .detach()
                .cpu()
            )
            correct += int((logits.argmax(dim=1) == batch_labels).sum().detach().cpu())
    return {"loss": loss_sum / len(labels), "accuracy": correct / len(labels)}


def aggregate_client_split(
    clients: tuple[ClientData, ...], split: str
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    images = np.concatenate([getattr(client, f"{split}_images") for client in clients])
    labels = np.concatenate([getattr(client, f"{split}_labels") for client in clients])
    return images, labels


def sample_client_ids(
    n_clients: int, fraction: float, generator: np.random.Generator
) -> NDArray[np.int64]:
    selected_count = max(1, round(n_clients * fraction))
    return np.sort(generator.choice(n_clients, size=selected_count, replace=False))


def plot_history(history: pd.DataFrame, output_prefix: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(history["round"], history["train_loss"], label="Training")
    axes[0].plot(history["round"], history["val_loss"], label="Validation")
    axes[0].set(xlabel="Communication round", ylabel="Cross-entropy loss", title="Loss")
    axes[1].plot(history["round"], history["train_accuracy"], label="Training")
    axes[1].plot(history["round"], history["val_accuracy"], label="Validation")
    axes[1].set(xlabel="Communication round", ylabel="Accuracy", title="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def train_fedavg(
    dataset: FederatedDataset,
    config: FedAvgConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """Train FedAvg and persist complete per-round metrics and selected clients."""

    config.validate(dataset.n_clients)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    generator = np.random.default_rng(config.seed)
    model = TwoLayerNet().to(device)
    global_state = clone_state(model.state_dict())
    train_images, train_labels = aggregate_client_split(dataset.clients, "train")
    val_images, val_labels = aggregate_client_split(dataset.clients, "val")
    records: list[dict[str, float | int]] = []
    selection_rows: list[dict[str, int]] = []
    started = time.perf_counter()

    def record_metrics(round_index: int, local_loss: float, local_accuracy: float) -> None:
        model.load_state_dict(global_state)
        train_metrics = evaluate_arrays(model, train_images, train_labels, device)
        val_metrics = evaluate_arrays(model, val_images, val_labels, device)
        records.append(
            {
                "round": round_index,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "mean_selected_local_loss": local_loss,
                "mean_selected_local_accuracy": local_accuracy,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    record_metrics(0, float("nan"), float("nan"))
    for round_index in range(1, config.rounds + 1):
        selected = sample_client_ids(dataset.n_clients, config.client_fraction, generator)
        states: list[StateDict] = []
        sample_counts: list[int] = []
        local_metrics: list[dict[str, float]] = []
        for client_id in selected:
            state, metrics = train_local_model(
                global_state,
                dataset.clients[int(client_id)],
                config,
                int(client_id),
                round_index,
                device,
            )
            states.append(state)
            sample_counts.append(dataset.clients[int(client_id)].train_size)
            local_metrics.append(metrics)
            selection_rows.append({"round": round_index, "client_id": int(client_id)})
        global_state = aggregate_states(states, sample_counts)
        record_metrics(
            round_index,
            float(np.mean([metrics["local_loss"] for metrics in local_metrics])),
            float(np.mean([metrics["local_accuracy"] for metrics in local_metrics])),
        )
        row = records[-1]
        print(
            f"round={round_index} train_acc={row['train_accuracy']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )

    model.load_state_dict(global_state)
    test_metrics = evaluate_arrays(model, dataset.test_images, dataset.test_labels, device)
    history = pd.DataFrame(records)
    history.to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_clients.csv", index=False)
    plot_history(history, run_dir / "training_curves", "FedAvg training")
    torch.save(global_state, run_dir / "final_model.pt")
    summary: dict[str, Any] = {
        "device": str(device),
        "rounds": config.rounds,
        "clients_per_round": max(1, round(dataset.n_clients * config.client_fraction)),
        "final_train_loss": float(history.iloc[-1]["train_loss"]),
        "final_train_accuracy": float(history.iloc[-1]["train_accuracy"]),
        "final_validation_loss": float(history.iloc[-1]["val_loss"]),
        "final_validation_accuracy": float(history.iloc[-1]["val_accuracy"]),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
