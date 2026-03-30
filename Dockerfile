# Base image (Python + PyTorch + CUDA pre-installed)
ARG PYTORCH_TAG=2.7.1-cuda11.8-cudnn9-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

WORKDIR /workspace

# System dependencies (opencv-python requires)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY . .

# Pre-download LPIPS VGG weights
RUN python -c "\
from taming.modules.losses.lpips import LPIPS; \
model = LPIPS(); \
print('LPIPS VGG weights downloaded')" \
    || echo "WARNING: LPIPS pre-download skipped"

# Environment
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py", "--help"]
