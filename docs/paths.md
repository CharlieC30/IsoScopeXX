# Path Assembly Reference

All paths are assembled from `cfg/env.json` (per-machine paths) and `cfg/{experiment}.yaml` (training parameters).

## Data Paths (Input)

```
$DATASET
└── {dataset}/
    ├── train/
    │   ├── {dir0}/
    │   └── {dir1}/
    └── val/
        ├── {dir0}/
        └── {dir1}/
```

`direction` is split by `_` to support paired directories:
- `x3d0`: single directory `train/x3d0/`
- `x3d0_x3d1`: paired directories `train/x3d0/` + `train/x3d1/`

Only `direction` uses `_` for splitting. Other parameters like `dataset` and `prj` are used as-is.

## Log Paths (Output)

```
$LOGS/
├── mlflow/
│   └── mlflow.db
│
└── {dataset}/
    └── {prj}/
        ├── logs/
        │   └── TensorBoardLogger/{run_timestamp}/
        └── checkpoints/
            └── {run_timestamp}/
                ├── config.json
                ├── {yaml_name}.yaml
                ├── {models}.py
                ├── {netg}_model_epoch_{N}.pth
                └── {netd}_model_epoch_{N}.pth
```

`{run_timestamp}` = `YYYYMMDD_HHMMSS`, shared across TensorBoard and checkpoints.

## Docker

With `--env docker`, paths resolve from the `docker` entry in `cfg/env.json`.
Volume mounts are defined in `docker-compose.yaml`:

| Host path (from `.env`) | Container path | Purpose |
|-------------------------|----------------|---------|
| `$DATA_PATH` | `/workspace/data` | Training datasets |
| `$LOGS_PATH` | `/workspace/logs` | MLflow, TensorBoard, checkpoints |

MLflow tracking URI inside the container: `http://mlflow:5002` (Docker internal DNS).
