"""Ray Core Actor implementation for parallel client updates."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
import torch

from .data import ClientData, FederatedDataset
from .fedavg import (
    FedAvgConfig,
    StateDict,
    aggregate_client_split,
    aggregate_states,
    clone_state,
    evaluate_arrays,
    plot_history,
    resolve_device,
    seed_everything,
    train_local_model,
)
from .model import TwoLayerNet


class ClientActor:
    """One short-lived Actor that owns exactly one selected client's local data."""

    def __init__(self, client_id: int, client: ClientData, config: FedAvgConfig) -> None:
        self.client_id = client_id
        self.client = client
        self.config = config

    def train(self, global_state: StateDict, round_index: int) -> tuple[StateDict, dict[str, float], int]:
        state, metrics = train_local_model(
            global_state,
            self.client,
            self.config,
            self.client_id,
            round_index,
            torch.device("cpu"),
        )
        return state, metrics, self.client.train_size


def train_ray_fedavg(
    dataset: FederatedDataset,
    config: FedAvgConfig,
    run_dir: Path,
    clients_per_round: int = 4,
    ray_cpus: int = 4,
) -> dict[str, Any]:
    """Run four client updates in parallel and recreate Actors every round."""

    if clients_per_round != 4:
        raise ValueError("the assignment requires exactly four clients per Ray round")
    if ray_cpus < 1:
        raise ValueError("ray_cpus must be positive")
    config.validate(dataset.n_clients)
    run_dir.mkdir(parents=True, exist_ok=True)
    ray_config = {
        **asdict(config),
        "clients_per_round": clients_per_round,
        "ray_cpus": ray_cpus,
        "actor_num_cpus": 1,
        "actor_num_gpus": 0,
    }
    (run_dir / "config.json").write_text(json.dumps(ray_config, indent=2), encoding="utf-8")
    seed_everything(config.seed)
    evaluation_device = resolve_device(config.device)
    generator = np.random.default_rng(config.seed)
    model = TwoLayerNet().to(evaluation_device)
    global_state = clone_state(model.state_dict())
    train_images, train_labels = aggregate_client_split(dataset.clients, "train")
    val_images, val_labels = aggregate_client_split(dataset.clients, "val")
    records: list[dict[str, float | int]] = []
    selections: list[dict[str, int]] = []
    started = time.perf_counter()

    def record(round_index: int, local_loss: float, local_accuracy: float) -> None:
        model.load_state_dict(global_state)
        train_metrics = evaluate_arrays(model, train_images, train_labels, evaluation_device)
        val_metrics = evaluate_arrays(model, val_images, val_labels, evaluation_device)
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

    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        ray.init(num_cpus=ray_cpus, include_dashboard=False, log_to_driver=False)
    actor_type = ray.remote(num_cpus=1, num_gpus=0)(ClientActor)
    try:
        record(0, float("nan"), float("nan"))
        for round_index in range(1, config.rounds + 1):
            selected = np.sort(
                generator.choice(dataset.n_clients, size=clients_per_round, replace=False)
            )
            actors = [
                actor_type.remote(int(client_id), dataset.clients[int(client_id)], config)
                for client_id in selected
            ]
            futures = [actor.train.remote(global_state, round_index) for actor in actors]
            results = ray.get(futures)
            for actor in actors:
                ray.kill(actor, no_restart=True)
            states = [result[0] for result in results]
            metrics = [result[1] for result in results]
            counts = [result[2] for result in results]
            global_state = aggregate_states(states, counts)
            for client_id in selected:
                selections.append({"round": round_index, "client_id": int(client_id)})
            record(
                round_index,
                float(np.mean([metric["local_loss"] for metric in metrics])),
                float(np.mean([metric["local_accuracy"] for metric in metrics])),
            )
            row = records[-1]
            print(
                f"ray_round={round_index} train_acc={row['train_accuracy']:.4f} "
                f"val_acc={row['val_accuracy']:.4f} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    finally:
        if started_ray_here:
            ray.shutdown()

    model.load_state_dict(global_state)
    test_metrics = evaluate_arrays(
        model, dataset.test_images, dataset.test_labels, evaluation_device
    )
    history = pd.DataFrame(records)
    history.to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(selections).to_csv(run_dir / "selected_clients.csv", index=False)
    plot_history(history, run_dir / "training_curves", "Ray Actor FedAvg training")
    torch.save(global_state, run_dir / "final_model.pt")
    summary: dict[str, Any] = {
        "evaluation_device": str(evaluation_device),
        "rounds": config.rounds,
        "clients_per_round": clients_per_round,
        "ray_cpus": ray_cpus,
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
