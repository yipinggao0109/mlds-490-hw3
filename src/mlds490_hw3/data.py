"""Loading and deterministic client-level splitting for Federated EMNIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClientData:
    """One client's private training and validation arrays."""

    train_images: NDArray[np.float32]
    train_labels: NDArray[np.int64]
    val_images: NDArray[np.float32]
    val_labels: NDArray[np.int64]

    @property
    def train_size(self) -> int:
        return len(self.train_labels)

    @property
    def val_size(self) -> int:
        return len(self.val_labels)


@dataclass(frozen=True)
class FederatedDataset:
    clients: tuple[ClientData, ...]
    test_images: NDArray[np.float32]
    test_labels: NDArray[np.int64]

    @property
    def n_clients(self) -> int:
        return len(self.clients)


def _validate_images_labels(images: NDArray, labels: NDArray, name: str) -> None:
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"{name} images must have shape (n, 28, 28), received {images.shape}")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError(f"{name} images and labels do not align")
    if len(labels) == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(images).all():
        raise ValueError(f"{name} contains non-finite pixels")
    if labels.min() < 0 or labels.max() >= 62:
        raise ValueError(f"{name} contains labels outside [0, 61]")


def load_federated_data(
    data_dir: Path,
    validation_fraction: float = 0.2,
    split_seed: int = 490,
) -> FederatedDataset:
    """Load Canvas data and split every client without touching held-out test data."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    train_path = data_dir / "train_data.npy"
    test_path = data_dir / "test_data.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("train_data.npy and test_data.npy are both required")

    raw_clients = np.load(train_path, allow_pickle=True)
    raw_test = np.load(test_path, allow_pickle=True)
    if raw_clients.shape != (100,):
        raise ValueError(f"expected 100 clients, received {raw_clients.shape}")
    if raw_test.shape != (1,):
        raise ValueError(f"expected one aggregated test object, received {raw_test.shape}")

    clients: list[ClientData] = []
    for client_id, record in enumerate(raw_clients):
        if not isinstance(record, dict) or not {"images", "labels"}.issubset(record):
            raise ValueError(f"client {client_id} has an invalid record")
        images = np.asarray(record["images"], dtype=np.float32)
        labels = np.asarray(record["labels"], dtype=np.int64)
        _validate_images_labels(images, labels, f"client {client_id}")

        generator = np.random.default_rng(split_seed + client_id)
        permutation = generator.permutation(len(labels))
        val_size = max(1, round(validation_fraction * len(labels)))
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        if len(train_indices) == 0:
            raise ValueError(f"client {client_id} has no training samples after splitting")
        clients.append(
            ClientData(
                train_images=np.ascontiguousarray(images[train_indices]),
                train_labels=np.ascontiguousarray(labels[train_indices]),
                val_images=np.ascontiguousarray(images[val_indices]),
                val_labels=np.ascontiguousarray(labels[val_indices]),
            )
        )

    test_record = raw_test[0]
    if not isinstance(test_record, dict) or not {"images", "labels"}.issubset(test_record):
        raise ValueError("test data has an invalid record")
    test_images = np.asarray(test_record["images"], dtype=np.float32)
    test_labels = np.asarray(test_record["labels"], dtype=np.int64)
    _validate_images_labels(test_images, test_labels, "test data")
    if set(np.concatenate([client.train_labels for client in clients])) != set(range(62)):
        raise ValueError("training split does not contain all 62 classes")
    if set(test_labels) != set(range(62)):
        raise ValueError("test data does not contain all 62 classes")
    return FederatedDataset(tuple(clients), test_images, test_labels)
