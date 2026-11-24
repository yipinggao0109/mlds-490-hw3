#!/usr/bin/env python3
"""Run FedAvg across several Laplace noise scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mlds490_hw3.data import load_federated_data
from mlds490_hw3.fedavg import FedAvgConfig, train_fedavg
from mlds490_hw3.plots import plot_dp_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Assignment3-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/differential_privacy"))
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_federated_data(args.data_dir, split_seed=args.seed)
    rows = []
    histories: dict[float, pd.DataFrame] = {}
    for scale in args.noise_scales:
        config = FedAvgConfig(
            rounds=args.rounds,
            client_fraction=0.10,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            noise_scale=scale,
        )
        run_dir = args.output_dir / f"b{scale:g}_seed{args.seed}"
        summary_path = run_dir / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else train_fedavg(dataset, config, run_dir)
        )
        rows.append({"noise_scale": scale, **summary})
        histories[scale] = pd.read_csv(run_dir / "history.csv")
    frame = pd.DataFrame(rows).sort_values("noise_scale")
    frame.to_csv(args.output_dir / "comparison.csv", index=False)
    plot_dp_comparison(frame, histories, args.output_dir / "comparison")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
