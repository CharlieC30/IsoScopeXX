# Docker

## Overview

The Docker setup provides two services managed by Compose profiles:

- **mlflow** (profile: `server`): MLflow tracking server with SQLite backend
- **training** (profile: `train`): GPU-enabled training container with PyTorch

Each profile can be started independently. See Deployment for usage patterns.

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) with Compose V2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver compatible with the target CUDA version

<!-- ```bash
docker run --rm --gpus all nvidia-smi
``` -->

## Quick Start

```bash
cp .env.example .env
# edit .env with your paths and settings
```

See Deployment for how to start each service.

MLflow UI: http://localhost:5003

## Deployment

Images are built automatically on first run. To rebuild after changing the Dockerfile or dependencies, see Configuration.

### A. MLflow + Training on Same Machine

```bash
docker compose --profile server up -d
docker compose --profile train run --rm training
```

### B. MLflow Server Only

```bash
docker compose --profile server up -d
```

Accessible at `http://<server-ip>:5003`.

### C. Training Only (Remote MLflow)

Set the remote URI in `.env`:

```env
EXTRA_TRAIN_ARGS=--tracking_uri http://<mlflow-server-ip>:5003
```

```bash
docker compose --profile train run --rm training
```

## Configuration

All settings are in `.env`. See `.env.example` for available variables and defaults.

To change CUDA version, edit `PYTORCH_TAG` and rebuild:

```bash
docker compose build training
```

To select specific GPUs:

```bash
NVIDIA_VISIBLE_DEVICES=0,1 docker compose --profile train run --rm training
```

## Security

The MLflow server uses `--allowed-hosts` to control which hosts can connect. The default allows `localhost` and Docker internal DNS (`mlflow`).

For remote access from other machines, set `MLFLOW_ALLOWED_HOSTS` in `.env`. This value **replaces** the default, so include all hosts:

```env
MLFLOW_ALLOWED_HOSTS=localhost:*,mlflow:*,<your-server-ip>:*
```

The MLflow server has no authentication. Use within a trusted network only.

## Cleanup

```bash
# Stop MLflow server
docker compose --profile server down

# Remove leftover training containers
docker compose rm -f training
```

## Notes

- Host port is 5003, mapped from container port 5002. To use a different host port, edit the mapping in `docker-compose.yaml` (e.g., `"5004:5002"`).
- Switching experiments does not require restarting MLflow or rebuilding. Edit `.env` and run training again.
- `out/` is inside the container and does not persist. Checkpoints and logs are saved to mounted volumes.
