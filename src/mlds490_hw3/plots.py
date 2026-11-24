"""Assignment-specific figures derived from experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import FederatedDataset


def plot_class_distributions(
    dataset: FederatedDataset,
    output_prefix: Path,
    client_ids: tuple[int, ...] = (0, 20, 40, 60, 80),
) -> None:
    if len(client_ids) != 5:
        raise ValueError("exactly five clients are required")
    rows = []
    all_labels = np.concatenate(
        [np.concatenate([client.train_labels, client.val_labels]) for client in dataset.clients]
    )
    rows.append(np.bincount(all_labels, minlength=62))
    for client_id in client_ids:
        client = dataset.clients[client_id]
        labels = np.concatenate([client.train_labels, client.val_labels])
        rows.append(np.bincount(labels, minlength=62))
    matrix = np.asarray(rows, dtype=np.float64)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 6.5),
        gridspec_kw={"height_ratios": [1.05, 2.0]},
        constrained_layout=True,
    )
    axes[0].bar(np.arange(62), matrix[0], color="#35618D", width=0.85)
    axes[0].set(
        xlabel="Class label",
        ylabel="Number of samples",
        title="Overall Federated EMNIST class distribution",
        xlim=(-0.8, 61.8),
    )
    axes[0].grid(axis="y", alpha=0.2)
    image = axes[1].imshow(normalized[1:], aspect="auto", cmap="magma")
    axes[1].set_yticks(np.arange(5), [f"Client {client_id}" for client_id in client_ids])
    axes[1].set_xticks(np.arange(0, 62, 5))
    axes[1].set(
        xlabel="Class label",
        ylabel="Selected client",
        title="Within-client class proportions for five clients",
    )
    fig.colorbar(image, ax=axes[1], label="Fraction of client samples")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)

def plot_hyperparameter_comparison(summary: pd.DataFrame, output_prefix: Path) -> None:
    labels = [f"C={100*c:.0f}%, E={e:d}" for c, e in zip(summary.client_fraction, summary.local_epochs)]
    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, summary.final_train_accuracy, width, label="Training")
    ax.bar(x + width / 2, summary.final_validation_accuracy, width, label="Validation")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="Final accuracy", title="FedAvg hyperparameter comparison", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_dp_comparison(
    summaries: pd.DataFrame,
    histories: dict[float, pd.DataFrame],
    output_prefix: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for scale, history in sorted(histories.items()):
        axes[0].plot(
            history["round"],
            history["train_accuracy"],
            label=f"b={scale:g} train",
            linewidth=1.5,
        )
        axes[0].plot(
            history["round"],
            history["val_accuracy"],
            linestyle="--",
            label=f"b={scale:g} validation",
            linewidth=1.2,
        )
    axes[0].set(
        xlabel="Communication round",
        ylabel="Accuracy",
        title="Accuracy during noisy FedAvg training",
        ylim=(0, 1),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(
        summaries.noise_scale,
        summaries.final_train_accuracy,
        marker="o",
        label="Final training",
    )
    axes[1].plot(
        summaries.noise_scale,
        summaries.test_accuracy,
        marker="s",
        label="Held-out test",
    )
    axes[1].set(
        xlabel="Laplace noise scale b",
        ylabel="Accuracy",
        title="Final accuracy versus noise scale",
        ylim=(0, 1),
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)
