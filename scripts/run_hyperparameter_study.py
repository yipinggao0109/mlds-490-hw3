#!/usr/bin/env python3
"""Compare client fractions and local epoch counts for standard FedAvg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mlds490_hw3.data import load_federated_data
from mlds490_hw3.fedavg import FedAvgConfig, train_fedavg
from mlds490_hw3.plots import plot_hyperparameter_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Assignment3-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hyperparameter_study"))
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_federated_data(args.data_dir, split_seed=args.seed)
    configurations = ((0.05, 1), (0.05, 5), (0.10, 1), (0.10, 5))
    rows = []
    for fraction, epochs in configurations:
        config = FedAvgConfig(
            rounds=args.rounds,
            client_fraction=fraction,
            local_epochs=epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        run_dir = args.output_dir / f"c{fraction:g}_e{epochs}_seed{args.seed}"
        summary_path = run_dir / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else train_fedavg(dataset, config, run_dir)
        )
        rows.append({"client_fraction": fraction, "local_epochs": epochs, **summary})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "comparison.csv", index=False)
    plot_hyperparameter_comparison(frame, args.output_dir / "comparison")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
