from pathlib import Path

import numpy as np
import pytest
import torch

from mlds490_hw3.data import ClientData, load_federated_data
from mlds490_hw3.fedavg import (
    FedAvgConfig,
    aggregate_states,
    perturb_images,
    sample_client_ids,
)
from mlds490_hw3.model import TwoLayerNet


def test_model_has_assignment_architecture() -> None:
    model = TwoLayerNet()
    linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    assert len(linear_layers) == 2
    assert linear_layers[0].in_features == 784
    assert linear_layers[0].out_features == 128
    assert linear_layers[1].out_features == 62
    assert model(torch.zeros(3, 28, 28)).shape == (3, 62)


def test_weighted_fedavg_is_exact() -> None:
    states = [
        {"weight": torch.tensor([1.0, 3.0])},
        {"weight": torch.tensor([5.0, 7.0])},
    ]
    result = aggregate_states(states, [1, 3])
    torch.testing.assert_close(result["weight"], torch.tensor([4.0, 6.0]))


def test_fedavg_is_permutation_invariant() -> None:
    states = [
        {"weight": torch.tensor([1.0])},
        {"weight": torch.tensor([4.0])},
        {"weight": torch.tensor([8.0])},
    ]
    forward = aggregate_states(states, [2, 3, 5])
    backward = aggregate_states(states[::-1], [5, 3, 2])
    torch.testing.assert_close(forward["weight"], backward["weight"])


def test_aggregate_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        aggregate_states([], [])
    with pytest.raises(ValueError):
        aggregate_states([{"x": torch.ones(1)}], [0])


def test_client_sampling_has_no_duplicates() -> None:
    selected = sample_client_ids(100, 0.10, np.random.default_rng(490))
    assert len(selected) == 10
    assert len(np.unique(selected)) == 10
    assert np.all(np.diff(selected) > 0)


def test_laplace_mechanism_is_reproducible_and_scale_zero_is_identity() -> None:
    images = np.zeros((4, 28, 28), dtype=np.float32)
    first = perturb_images(images, 0.1, seed=1)
    second = perturb_images(images, 0.1, seed=1)
    np.testing.assert_array_equal(first, second)
    assert perturb_images(images, 0.0, seed=1) is images


def test_laplace_empirical_moments_match_theory() -> None:
    images = np.zeros((400, 28, 28), dtype=np.float32)
    noisy = perturb_images(images, 0.2, seed=2)
    assert float(noisy.mean()) == pytest.approx(0.0, abs=0.003)
    assert float(noisy.var()) == pytest.approx(2 * 0.2**2, rel=0.03)


def test_config_enforces_client_fraction_limit() -> None:
    with pytest.raises(ValueError):
        FedAvgConfig(client_fraction=0.11).validate(100)


def test_client_data_sizes() -> None:
    client = ClientData(
        np.zeros((8, 28, 28), dtype=np.float32),
        np.zeros(8, dtype=np.int64),
        np.zeros((2, 28, 28), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
    )
    assert client.train_size == 8
    assert client.val_size == 2


def test_canvas_dataset_loads_and_splits() -> None:
    data_dir = Path(__file__).parents[1] / "Assignment3-data"
    if not data_dir.exists():
        pytest.skip("Canvas data is intentionally not tracked by Git")
    dataset = load_federated_data(data_dir)
    assert dataset.n_clients == 100
    assert sum(client.train_size + client.val_size for client in dataset.clients) == 31_825
    assert len(dataset.test_labels) == 3_621
    assert set(dataset.test_labels) == set(range(62))
