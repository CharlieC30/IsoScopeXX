# Docker Usage

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver compatible with your target CUDA version

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

```bash
# 1. Create your .env file from the template
cp .env.example .env

# 2. Edit .env with your machine's paths and experiment settings

# 3. Build the training image
docker compose build training

# 4. Start MLflow tracking server
docker compose up mlflow -d

# 5. Run training
docker compose --profile train run training
```

## Common Operations

### View MLflow UI (without training)

```bash
docker compose up mlflow
# Open http://localhost:5002 in your browser
```

### Run training

```bash
docker compose --profile train run training
```

### Pass additional training arguments

Option A — set `EXTRA_TRAIN_ARGS` in `.env`:

```env
EXTRA_TRAIN_ARGS=--nocut --downbranch 2 --cropz 32
```

Option B — override the command directly:

```bash
docker compose run training python train.py \
    --yaml aisr --prj my_experiment --env docker \
    --nocut --downbranch 2 --cropz 32
```

### Switch experiments

When changing `DATASET_NAME` or `PRJ`, restart the MLflow service:

```bash
# 1. Edit .env with new DATASET_NAME and/or PRJ
# 2. Restart MLflow (its DB path depends on these values)
docker compose down mlflow
docker compose up mlflow -d
# 3. Run training
docker compose --profile train run training
```

### View training logs

```bash
# Follow logs in real time
docker compose logs -f training

# TensorBoard (from host)
tensorboard --logdir ${LOGS_PATH}/${DATASET_NAME}/${PRJ}/logs/TensorBoardLogger
```

## CUDA Version

The default build uses `pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime`. To use a different CUDA version, edit `PYTORCH_TAG` in `.env`:

```env
# CUDA 11.8 (default)
PYTORCH_TAG=2.7.1-cuda11.8-cudnn9-runtime

# CUDA 12.8
PYTORCH_TAG=2.8.0-cuda12.8-cudnn9-runtime
```

See all available tags at https://hub.docker.com/r/pytorch/pytorch/tags.

Then rebuild:

```bash
docker compose build training
```

## Multi-GPU

By default, all GPUs are available to the training container. To limit GPU access:

```bash
NVIDIA_VISIBLE_DEVICES=0,1 docker compose --profile train run training
```

## Architecture

```
docker-compose.yaml
├── mlflow service        (python:3.10-slim, long-running)
│   ├── SQLite DB at      /workspace/logs/{dataset}/{prj}/logs/mlflow.db
│   ├── Artifacts at      /workspace/logs/{dataset}/{prj}/logs/mlartifacts/
│   └── Port 5002
└── training service      (pytorch/pytorch, run-and-stop)
    ├── Data from         /workspace/data  ← volume mount
    ├── Logs to           /workspace/logs  ← volume mount
    └── --env docker      (reads cfg/env.json "docker" entry)
```

## Notes

- The `out/` directory (debug TIF images) is inside the container and does not persist after the container stops. Checkpoints and logs are saved to mounted volumes and are not affected.
- Git hash in MLflow tags will show `unknown` inside the container.
- The training image is approximately 8-12 GB due to CUDA and PyTorch.
