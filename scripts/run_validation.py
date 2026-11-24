"""Run quantitative checks that go beyond the unit test suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mlds490_hw3.data import load_federated_data
from mlds490_hw3.fedavg import (
    FedAvgConfig,
    aggregate_states,
    clone_state,
    perturb_images,
    sample_client_ids,
    train_local_model,
)
from mlds490_hw3.model import TwoLayerNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Assignment3-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    return parser.parse_args()


def max_state_difference(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
    return max(float((first[key] - second[key]).abs().max()) for key in first)


def image_hashes(images: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_federated_data(args.data_dir)

    # Weighted aggregation is checked against an independently computed reference.
    generator = np.random.default_rng(490)
    states = [
        {"weight": torch.tensor(generator.normal(size=20), dtype=torch.float32)}
        for _ in range(7)
    ]
    counts = [13, 29, 41, 7, 53, 11, 37]
    aggregated = aggregate_states(states, counts)["weight"].numpy()
    reference = np.average(
        np.stack([state["weight"].numpy() for state in states]), axis=0, weights=counts
    )
    aggregation_error = float(np.max(np.abs(aggregated - reference)))

    # With learning rate zero, every local state must equal the starting state exactly.
    torch.manual_seed(490)
    initial_state = clone_state(TwoLayerNet().state_dict())
    zero_config = FedAvgConfig(
        rounds=1,
        client_fraction=0.10,
        local_epochs=3,
        learning_rate=0.0,
        batch_size=31,
        device="cpu",
    )
    unchanged_state, _ = train_local_model(
        initial_state, dataset.clients[0], zero_config, 0, 1, torch.device("cpu")
    )
    zero_lr_error = max_state_difference(initial_state, unchanged_state)

    # Repeated sampling should be reasonably balanced and never select a client twice.
    sampling_generator = np.random.default_rng(490)
    participation = np.zeros(dataset.n_clients, dtype=np.int64)
    duplicate_rounds = 0
    for _ in range(2_000):
        selected = sample_client_ids(dataset.n_clients, 0.10, sampling_generator)
        duplicate_rounds += int(len(selected) != len(np.unique(selected)))
        participation[selected] += 1
    participation_cv = float(participation.std(ddof=1) / participation.mean())

    # Check the assignment's Laplace mechanism against its theoretical moments.
    scales = np.array([0.05, 0.10, 0.20, 0.40])
    noise_rows: list[dict[str, float]] = []
    zeros = np.zeros((2_000, 28, 28), dtype=np.float32)
    for index, scale in enumerate(scales):
        noise = perturb_images(zeros, float(scale), seed=20_000 + index)
        empirical_mean = float(noise.mean())
        empirical_variance = float(noise.var())
        theoretical_variance = float(2 * scale**2)
        noise_rows.append(
            {
                "scale": float(scale),
                "empirical_mean": empirical_mean,
                "empirical_variance": empirical_variance,
                "theoretical_variance": theoretical_variance,
                "relative_variance_error": abs(empirical_variance - theoretical_variance)
                / theoretical_variance,
            }
        )

    # Exact byte-level matches would indicate direct train/test duplication.
    train_images = np.concatenate(
        [np.concatenate((client.train_images, client.val_images)) for client in dataset.clients]
    )
    overlap_count = len(image_hashes(train_images) & image_hashes(dataset.test_images))
    overlap_fraction = overlap_count / len(dataset.test_images)

    checks = {
        "weighted_aggregation_max_abs_error": aggregation_error,
        "zero_learning_rate_max_parameter_change": zero_lr_error,
        "sampling_duplicate_rounds_out_of_2000": duplicate_rounds,
        "participation_min": int(participation.min()),
        "participation_max": int(participation.max()),
        "participation_mean": float(participation.mean()),
        "participation_coefficient_of_variation": participation_cv,
        "exact_train_test_image_overlap": overlap_count,
        "exact_train_test_image_overlap_fraction": overlap_fraction,
        "max_laplace_relative_variance_error": max(
            row["relative_variance_error"] for row in noise_rows
        ),
    }
    thresholds = {
        "aggregation_error_below_1e-6": aggregation_error < 1e-6,
        "zero_lr_is_exactly_unchanged": zero_lr_error == 0.0,
        "sampling_has_no_duplicates": duplicate_rounds == 0,
        "participation_cv_below_0.10": participation_cv < 0.10,
        "provided_data_overlap_below_one_percent": overlap_fraction < 0.01,
        "laplace_variance_error_below_0.02": checks[
            "max_laplace_relative_variance_error"
        ]
        < 0.02,
    }
    output = {"measurements": checks, "checks": thresholds, "all_checks_passed": all(thresholds.values())}
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    pd.DataFrame(noise_rows).to_csv(args.output_dir / "laplace_moments.csv", index=False)
    pd.DataFrame(
        {"client_id": np.arange(dataset.n_clients), "participation_count": participation}
    ).to_csv(args.output_dir / "participation_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].bar(np.arange(dataset.n_clients), participation, width=0.9, color="#4c78a8")
    axes[0].axhline(participation.mean(), color="#e45756", linestyle="--", label="Mean")
    axes[0].set(
        xlabel="Client ID",
        ylabel="Selections in 2,000 rounds",
        title="Client participation audit",
    )
    axes[0].legend()
    axes[1].plot(scales, 2 * scales**2, marker="o", label="Theoretical $2b^2$")
    axes[1].scatter(
        scales,
        [row["empirical_variance"] for row in noise_rows],
        marker="x",
        s=70,
        label="Empirical",
    )
    axes[1].set(
        xlabel="Laplace scale b",
        ylabel="Noise variance",
        title="Mechanism moment audit",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_audit.png", dpi=200)
    fig.savefig(args.output_dir / "validation_audit.pdf")
    plt.close(fig)

    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit("one or more validation checks failed")


if __name__ == "__main__":
    main()
