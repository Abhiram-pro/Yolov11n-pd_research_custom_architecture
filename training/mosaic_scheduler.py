"""
mosaic_scheduler.py

Two-phase training helper that reproduces the paper's "Mosaic only in final 10 epochs" trick.

Usage pattern (invoked by train_wrapper.py):
- Phase 1: train for epochs - mosaic_last_epochs with mosaic disabled
- Phase 2: resume training for mosaic_last_epochs with mosaic enabled

This script exposes `run_two_phase_training(config: dict)` which the wrapper uses.
"""

import os
import time
import shutil
import subprocess
from typing import Dict, Optional

# Try to import Ultralytics YOLO API. If not available, fallback to CLI via subprocess.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


def _call_cli_train(model_path: str, data_path: str, epochs: int, batch: int, imgsz: int,
                    optimizer: str, lr0: float, project: str, name: str, save_period: int,
                    resume: Optional[str], mosaic: bool, extra_opts: Optional[Dict] = None):
    """
    Fallback CLI trainer invocation (ultralytics 'yolo' CLI).
    Example CLI:
      yolo task=detect mode=train model=... data=... epochs=... imgsz=... batch=... optimizer=SGD lr0=0.01
    """
    cmd = [
        "yolo",
        "task=detect",
        "mode=train",
        f"model={model_path}",
        f"data={data_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"optimizer={optimizer}",
        f"lr0={lr0}",
        f"project={project}",
        f"name={name}",
    ]
    if resume:
        cmd.append(f"resume={resume}")
    # mosaic control: some loaders accept 'mosaic' as dataset/hyp flag; we pass as extra CLI arg
    cmd.append(f"mosaic={1 if mosaic else 0}")
    # add save_period if provided
    if save_period:
        cmd.append(f"save_period={save_period}")
    # append any extras
    if extra_opts:
        for k, v in extra_opts.items():
            cmd.append(f"{k}={v}")

    print("Running CLI command:", " ".join(cmd))
    subprocess.check_call(cmd)


def _api_train_phase(model_path: str, data_path: str, epochs: int, batch: int, imgsz: int,
                     optimizer: str, lr0: float, project: str, name: str, save_period: int,
                     resume: Optional[str], mosaic: bool, seed: Optional[int]):
    """
    Use the ultralytics YOLO Python API for training one phase.
    Note: 'mosaic' control depends on dataset/hyperparam handling by the YOLO API. We pass mosaic via
    'overrides' where supported. If not honored by your dataset pipeline, consider the CLI fallback.
    """
    print(f"ULTRALYTICS API training phase: epochs={epochs}, mosaic={mosaic}, resume={resume}")
    model = YOLO(model_path)
    # build args
    train_args = {
        "data": data_path,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "optimizer": optimizer,
        "lr0": lr0,
        "project": project,
        "name": name,
        "save_period": save_period,
        # the YOLO API accepts 'resume' as bool/str in some versions; set resume path via 'resume' if available
    }
    if resume:
        train_args["resume"] = resume
    # pass seed if provided
    if seed is not None:
        train_args["seed"] = seed

    # Many Ultralytics versions support 'overrides' or pass-through of arbitrary kwargs.
    # We'll attempt to set mosaic via 'mosaic' kwarg; if not supported, fallback to CLI.
    try:
        train_args["mosaic"] = bool(mosaic)
        print("Starting Ultralytics train with args:", train_args)
        result = model.train(**train_args)
        return result
    except TypeError as e:
        # API didn't accept mosaic / overrides; signal to fallback
        print("Ultralytics.train() did not accept mosaic override or args. Falling back to CLI. Error:", e)
        raise RuntimeError("Ultralytics API mosaic override not supported in this environment.")


def run_two_phase_training(config: Dict):
    """
    Run two-phase training using either Ultralytics API or CLI fallback.

    config keys expected (examples):
      - model: path to model YAML
      - data: path to data YAML
      - epochs: total epochs (int)
      - batch: batch size
      - imgsz: image size
      - optimizer: 'SGD'
      - lr0: 0.01
      - project: runs
      - name: run name
      - save_period: int or None
      - mosaic_last_epochs: 10
      - seed: random seed (optional)
    """
    # ensure required keys
    required = ["model", "data", "epochs", "batch", "imgsz", "optimizer", "lr0", "project", "name", "mosaic_last_epochs"]
    for k in required:
        if k not in config:
            raise ValueError(f"Missing required config key: {k}")

    total_epochs = int(config["epochs"])
    last_k = int(config["mosaic_last_epochs"])
    assert last_k >= 0 and last_k < total_epochs, "mosaic_last_epochs must be >=0 and < total_epochs"

    phase1_epochs = total_epochs - last_k
    phase2_epochs = last_k

    # directories
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['name']}_{timestamp}"
    run_dir = os.path.join(config.get("project", "runs"), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Phase 1: no mosaic
    print(f"Phase 1: training {phase1_epochs} epochs with mosaic=False")
    resume_ckpt = None
    try:
        if ULTRALYTICS_AVAILABLE:
            _api_train_phase(
                model_path=config["model"],
                data_path=config["data"],
                epochs=phase1_epochs,
                batch=config["batch"],
                imgsz=config["imgsz"],
                optimizer=config["optimizer"],
                lr0=config["lr0"],
                project=config["project"],
                name=run_name + "_phase1",
                save_period=config.get("save_period", 10),
                resume=None,
                mosaic=False,
                seed=config.get("seed", None)
            )
            # ultralytics saves best/latest checkpoint under runs/<run_name> by default.
            # We attempt to find the last checkpoint path.
            # Common pattern: runs/<run_name>/weights/last.pt
            possible = os.path.join(config.get("project", "runs"), run_name + "_phase1", "weights", "last.pt")
            if os.path.exists(possible):
                resume_ckpt = possible
        else:
            # CLI fallback: call subprocess
            _call_cli_train(
                model_path=config["model"],
                data_path=config["data"],
                epochs=phase1_epochs,
                batch=config["batch"],
                imgsz=config["imgsz"],
                optimizer=config["optimizer"],
                lr0=config["lr0"],
                project=config["project"],
                name=run_name + "_phase1",
                save_period=config.get("save_period", 10),
                resume=None,
                mosaic=False
            )
            possible = os.path.join(config.get("project", "runs"), run_name + "_phase1", "weights", "last.pt")
            if os.path.exists(possible):
                resume_ckpt = possible
    except Exception as e:
        print("Phase 1 training raised an exception:", e)
        print("Attempting to continue to Phase 2 without resume (not recommended).")
        resume_ckpt = None

    # Phase 2: final epochs with mosaic enabled
    if phase2_epochs > 0:
        print(f"Phase 2: training final {phase2_epochs} epochs with mosaic=True (resume from Phase1 checkpoint)")
        try:
            if ULTRALYTICS_AVAILABLE:
                _api_train_phase(
                    model_path=config["model"],
                    data_path=config["data"],
                    epochs=phase2_epochs,
                    batch=config["batch"],
                    imgsz=config["imgsz"],
                    optimizer=config["optimizer"],
                    lr0=config["lr0"],
                    project=config["project"],
                    name=run_name + "_phase2",
                    save_period=config.get("save_period", 10),
                    resume=resume_ckpt,
                    mosaic=True,
                    seed=config.get("seed", None)
                )
            else:
                _call_cli_train(
                    model_path=config["model"],
                    data_path=config["data"],
                    epochs=phase2_epochs,
                    batch=config["batch"],
                    imgsz=config["imgsz"],
                    optimizer=config["optimizer"],
                    lr0=config["lr0"],
                    project=config["project"],
                    name=run_name + "_phase2",
                    save_period=config.get("save_period", 10),
                    resume=resume_ckpt,
                    mosaic=True
                )
        except Exception as e:
            print("Phase 2 training failed:", e)
            raise

    print("Two-phase training finished. Check runs directory for outputs.")
    return run_dir
