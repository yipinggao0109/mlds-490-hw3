#!/usr/bin/env python3
"""Validate the Canvas dataset and plot overall and client class distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlds490_hw3.data import load_federated_data
from mlds490_hw3.plots import plot_class_distributions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Assignment3-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_analysis"))
    parser.add_argument("--seed", type=int, default=490)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_federated_data(args.data_dir, split_seed=args.seed)
    plot_class_distributions(dataset, args.output_dir / "class_distributions")
    client_sizes = [client.train_size + client.val_size for client in dataset.clients]
    summary = {
        "clients": dataset.n_clients,
        "total_local_samples": int(sum(client_sizes)),
        "minimum_client_samples": int(min(client_sizes)),
        "maximum_client_samples": int(max(client_sizes)),
        "mean_client_samples": float(np.mean(client_sizes)),
        "training_samples_after_split": int(sum(client.train_size for client in dataset.clients)),
        "validation_samples_after_split": int(sum(client.val_size for client in dataset.clients)),
        "test_samples": len(dataset.test_labels),
        "classes": len(np.unique(dataset.test_labels)),
        "selected_clients_for_plot": [0, 20, 40, 60, 80],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
