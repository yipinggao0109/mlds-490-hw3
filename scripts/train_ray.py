#!/usr/bin/env python3
"""Train FedAvg with four short-lived Ray Core Actors per round."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlds490_hw3.data import load_federated_data
from mlds490_hw3.fedavg import FedAvgConfig
from mlds490_hw3.ray_training import train_ray_fedavg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Assignment3-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ray"))
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ray-cpus", type=int, default=4)
    args = parser.parse_args()
    dataset = load_federated_data(args.data_dir, split_seed=args.seed)
    config = FedAvgConfig(
        rounds=args.rounds,
        client_fraction=0.04,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    train_ray_fedavg(
        dataset,
        config,
        args.output_dir / f"c4_e{args.local_epochs}_seed{args.seed}",
        clients_per_round=4,
        ray_cpus=args.ray_cpus,
    )


if __name__ == "__main__":
    main()
