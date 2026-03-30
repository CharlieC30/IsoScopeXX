# Docker

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
# Edit .env with your paths and settings

docker compose build training
docker compose --profile server up -d
docker compose --profile train run training
```

MLflow UI: http://localhost:5002

## Deployment

Both services use Compose profiles and require `--profile` to start.

### A. MLflow + Training on Same Machine

```bash
docker compose --profile server up -d
docker compose --profile train run training
```

### B. MLflow Server Only

```bash
docker compose --profile server up -d
```

Accessible at `http://<server-ip>:5002`.

### C. Training Only (Remote MLflow)

Set the remote URI in `.env`:

```env
EXTRA_TRAIN_ARGS=--tracking_uri http://<mlflow-server-ip>:5002
```

```bash
docker compose build training
docker compose --profile train run training
```

## Configuration

All settings are in `.env`. See `.env.example` for available variables and defaults.

To change CUDA version, edit `PYTORCH_TAG` and rebuild:

```bash
docker compose build training
```

To select specific GPUs:

```bash
NVIDIA_VISIBLE_DEVICES=0,1 docker compose --profile train run training
```

## Notes

- The MLflow server has no authentication. Use within a trusted network only.
- If port 5002 is already in use on the host, change the port mapping in `docker-compose.yaml`: `"5003:5002"`. The container-internal port (5002) stays the same.
- Switching experiments does not require restarting MLflow or rebuilding. Change `.env` and run training again.
- The `out/` directory is inside the container and does not persist after the container stops. Checkpoints and logs are saved to mounted volumes.
