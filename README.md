🚀 YOLO11n-PD

A custom lightweight, high-accuracy road damage detection model based on YOLO11 + PD modules

YOLO11n-PD is a custom object detection architecture inspired by YOLOv8-PD (pavement distress), rebuilt from scratch with modern modules and optimized for crack, pothole, and road-distress detection.

This repo includes:

✨ A full PyTorch custom YOLO11n-PD model

🧠 Modules: Ghost, C3k2Ghost, BOT, LSKA, LSCD-Head

🎯 Complete 2-phase training pipeline

📦 Ultralytics-style YAML model spec

🔧 Hooks for dataset conversion, inference, and ablations

🔥 Features
🧩 Custom Architecture

Built using new lightweight blocks:

GhostConv + GhostBottleneck — efficient feature extraction

C3k2Ghost — improved C3 block using Ghost

BOTBlock — Bottleneck Transformer for global attention

LSKA — Large-Separable-Kernel Attention (captures long thin cracks)

LSCDHead — Shared convolutional head (lighter & more accurate)

🎛 Two-Phase Training (Paper-Style)

Phase 1: No Mosaic

Phase 2 (last 10 epochs): Mosaic ON
Exactly matching the augmentation schedule from YOLOv8-PD.

⚙ Fully Modular

Pluggable modules in models/

Pure PyTorch forward pass

Compatible with Ultralytics loader (via YAML)

📁 Project Structure
yolo11n-pd/
│
├── models/
│   ├── ghost.py
│   ├── c3k2_ghost.py
│   ├── bot.py
│   ├── lska.py
│   ├── lscd_head.py
│   └── model.py   # full YOLO11n-PD assembly
│
├── training/
│   ├── train.yaml
│   ├── mosaic_scheduler.py
│   └── train_wrapper.py
│
├── data/
│   ├── rdd2022.yaml
│   └── rdd2022_to_yolo.py   # (optional) dataset converter
│
└── README.md

🚀 Quick Start
1. Clone the repository
git clone https://github.com/<your-username>/YOLO11n-PD.git
cd YOLO11n-PD

2. Install dependencies
pip install ultralytics
pip install torch torchvision

3. Train (2-phase pipeline)
python training/train_wrapper.py --config training/train.yaml


This will automatically run:

Phase 1 (mosaic off)

Phase 2 (last 10 epochs, mosaic on)

Checkpoints saved under runs/

📈 Inference Example

Pure PyTorch:

from models.model import build_yolo11n_pd
import torch

model = build_yolo11n_pd(num_classes=4)
x = torch.randn(1, 3, 640, 640)
preds = model(x)
print(preds)

🧪 Model Goals

YOLO11n-PD is designed for:

Crack detection

Long/linear defect detection

Potholes and small-scale road hazards

On-device low-latency inference (mobile/eGPU/drones)