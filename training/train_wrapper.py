"""
train_wrapper.py

Convenience script to launch two-phase training for YOLO11n-PD.

Usage:
    python train_wrapper.py --config training/train.yaml
    python train_wrapper.py --config training/train.yaml --epochs 200 --batch 32

The script will:
- parse YAML config
- optionally override hyperparams via CLI args
- set deterministic seed for reproducibility
- call run_two_phase_training(config)
"""

import os
import argparse
import yaml
import random
import numpy as np
import torch

from mosaic_scheduler import run_two_phase_training

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Torch deterministic flags (may slow training)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11n-PD with mosaic-last-epochs scheduler")
    parser.add_argument("--config", type=str, default="training/train.yaml", help="Path to train.yaml")
    parser.add_argument("--epochs", type=int, help="Override total epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--name", type=str, help="Override run name prefix")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--dryrun", action="store_true", help="Print planned commands but don't run")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # override from CLI
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz
    if args.name:
        cfg["name"] = args.name
    if args.seed:
        cfg["seed"] = args.seed

    # defaults if missing
    cfg.setdefault("project", "runs")
    cfg.setdefault("save_period", 10)
    cfg.setdefault("name", cfg.get("name_prefix", "yolo11n_pd_expt"))

    seed = int(cfg.get("seed", 42))
    print(f"Using seed: {seed}")
    set_seed(seed)

    print("Training config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if args.dryrun:
        print("Dry run mode - not starting training")
        return

    run_dir = run_two_phase_training(cfg)
    print("Training complete. Outputs in:", run_dir)


if __name__ == "__main__":
    main()
