# MLDS 490 Homework 3

This repository contains a from-scratch PyTorch implementation of Federated Averaging on the Federated EMNIST data supplied with the assignment. It also contains a Ray Core Actor implementation and experiments with input-level Laplace noise.

## Repository contents

- `src/mlds490_hw3/data.py`: data loading, validation, and deterministic 80/20 splits for every client
- `src/mlds490_hw3/model.py`: two-layer fully connected network with 128 hidden units and ReLU
- `src/mlds490_hw3/fedavg.py`: local training, sample-weighted FedAvg, evaluation, and Laplace perturbation
- `src/mlds490_hw3/ray_training.py`: four-client parallel training with short-lived Ray Actors
- `scripts/`: entry points for data analysis, experiments, and quantitative validation
- `tests/`: unit tests for the model, aggregation, client sampling, noise, and data loader
- `results/`: CSV and JSON measurements plus the figures used in the report

The Canvas data and trained model checkpoints are intentionally excluded from version control. Put `train_data.npy` and `test_data.npy` in `Assignment3-data/` before running the code.

## Environment

Python 3.12 was used for the reported experiments.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Reproduce the results

Run all commands from the repository root.

```bash
python scripts/analyze_data.py

python scripts/run_hyperparameter_study.py \
  --rounds 100 --fractions 0.05 0.10 --local-epochs 1 5 \
  --learning-rate 0.05 --device cpu

python scripts/train_fedavg.py \
  --rounds 500 --client-fraction 0.10 --local-epochs 5 \
  --learning-rate 0.05 --device cpu \
  --run-dir results/fedavg/selected_c0.1_e5_r500

python scripts/train_ray.py \
  --rounds 100 --local-epochs 5 --learning-rate 0.05 \
  --device cpu --run-dir results/ray/c4_e5_seed490

python scripts/run_dp_study.py \
  --rounds 300 --local-epochs 5 --learning-rate 0.05 --device cpu

python scripts/run_validation.py
pytest -q
ruff check .
```

The fixed random seed is 490. CPU execution was used for the reported runs because a short benchmark on this small network was faster than MPS. Ray uses four CPU workers, one CPU per Actor and no GPU allocation.

## Main results

The selected standard FedAvg run used 10 clients per round, 5 local epochs, and 500 communication rounds. It achieved 64.51% validation accuracy and 65.34% held-out test accuracy. The Ray experiment used four clients per round and achieved 45.10% held-out test accuracy after 100 rounds. The differential privacy study reports the full comparison for noise scales from 0 to 0.4 in `results/differential_privacy/comparison.csv`.
