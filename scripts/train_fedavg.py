#!/usr/bin/env python3
"""Train the standard FedAvg implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlds490_hw3.data import load_federated_data
from mlds490_hw3.fedavg import FedAvgConfig, train_fedavg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Assignment3-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fedavg"))
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--client-fraction", type=float, default=0.10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    dataset = load_federated_data(args.data_dir, split_seed=args.seed)
    config = FedAvgConfig(
        rounds=args.rounds,
        client_fraction=args.client_fraction,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        momentum=args.momentum,
        seed=args.seed,
        device=args.device,
        noise_scale=args.noise_scale,
    )
    run_name = args.run_name or (
        f"c{args.client_fraction:g}_e{args.local_epochs}_b{args.noise_scale:g}_seed{args.seed}"
    )
    train_fedavg(dataset, config, args.output_dir / run_name)


if __name__ == "__main__":
    main()
