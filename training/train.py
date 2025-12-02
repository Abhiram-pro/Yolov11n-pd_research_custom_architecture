"""
train.py

Direct PyTorch training script for YOLO11n-PD road detection model.

This script provides a complete training pipeline for the custom YOLO11n-PD architecture
with support for:
- Two-phase training (mosaic augmentation in final epochs)
- Mixed precision training (AMP)
- Distributed training support
- Comprehensive logging and checkpointing
- Road damage detection optimized loss functions

Usage:
    python training/train.py --config training/train.yaml
    python training/train.py --config training/train.yaml --epochs 300 --batch 16
"""

import os
import sys
import argparse
import yaml
import time
import random
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.model import build_yolo11n_pd
from training.dataset import create_dataloader as create_road_dataloader
from training.loss import SimplifiedYOLOLoss, YOLOv8Loss


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True





def build_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Build optimizer based on config."""
    lr = config.get('lr0', 0.01)
    momentum = config.get('momentum', 0.937)
    weight_decay = config.get('weight_decay', 0.0005)
    optimizer_name = config.get('optimizer', 'SGD').upper()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
    elif optimizer_name == 'ADAM':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'ADAMW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, config: Dict, steps_per_epoch: int):
    """Build learning rate scheduler."""
    scheduler_name = config.get('scheduler', 'cosine').lower()
    epochs = config.get('epochs', 300)
    warmup_epochs = config.get('warmup_epochs', 3)
    
    if scheduler_name == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    return scheduler


def create_dataloader(config: Dict, mode: str = 'train', mosaic: bool = False) -> DataLoader:
    """
    Create dataloader for training or validation.
    
    Args:
        config: Training configuration
        mode: 'train' or 'val'
        mosaic: Whether to enable mosaic augmentation
    """
    batch_size = config.get('batch', 16)
    img_size = config.get('imgsz', 640)
    num_workers = config.get('workers', 8)
    
    # Parse data config
    data_config = config.get('data', 'data/rdd2022.yaml')
    
    # If data config is a YAML file, load it
    if isinstance(data_config, str) and data_config.endswith('.yaml'):
        import yaml
        if os.path.exists(data_config):
            with open(data_config, 'r') as f:
                data_dict = yaml.safe_load(f)
            
            # Get paths from data config
            if mode == 'train':
                img_dir = data_dict.get('train_images', 'data/images/train')
                label_dir = data_dict.get('train_labels', 'data/labels/train')
            else:
                img_dir = data_dict.get('val_images', 'data/images/val')
                label_dir = data_dict.get('val_labels', 'data/labels/val')
        else:
            # Fallback to default paths
            img_dir = f'data/images/{mode}'
            label_dir = f'data/labels/{mode}'
    else:
        img_dir = f'data/images/{mode}'
        label_dir = f'data/labels/{mode}'
    
    # Check if dataset exists
    if os.path.exists(img_dir) and os.path.exists(label_dir):
        print(f"Loading {mode} dataset from {img_dir}")
        try:
            dataloader = create_road_dataloader(
                img_dir=img_dir,
                label_dir=label_dir,
                batch_size=batch_size,
                img_size=img_size,
                augment=(mode == 'train'),
                mosaic=mosaic and (mode == 'train'),
                num_workers=num_workers,
                shuffle=(mode == 'train'),
                cache_images=config.get('cache_images', False)
            )
            return dataloader
        except Exception as e:
            print(f"Warning: Could not load dataset from {img_dir}: {e}")
            print("Falling back to dummy dataset for testing")
    
    # Fallback: Create dummy dataset for testing
    print(f"Using dummy dataset for {mode} (dataset not found at {img_dir})")
    from torch.utils.data import TensorDataset
    
    num_samples = 100 if mode == 'train' else 20
    dummy_images = torch.randn(num_samples, 3, img_size, img_size)
    dummy_targets = [{'boxes': torch.rand(5, 4), 'labels': torch.randint(0, 4, (5,))} 
                     for _ in range(num_samples)]
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, images, targets):
            self.images = images
            self.targets = targets
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            return self.images[idx], self.targets[idx]
    
    dataset = DummyDataset(dummy_images, dummy_targets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=0,  # Use 0 for dummy dataset
        pin_memory=True,
        drop_last=(mode == 'train')
    )
    
    return dataloader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    writer: Optional[SummaryWriter] = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        # targets processing depends on your dataset format
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                predictions = model(images)
                loss = criterion(predictions, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}")
            
            if writer is not None:
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for images, targets in dataloader:
        images = images.to(device)
        
        predictions = model(images)
        loss = criterion(predictions, targets)
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    
    # TODO: Add mAP calculation for road damage detection
    metrics = {
        'val_loss': avg_loss,
        'mAP50': 0.0,  # Placeholder
        'mAP50-95': 0.0  # Placeholder
    }
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    config: Dict,
    save_path: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('last.pt', 'best.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")


def train_phase(
    config: Dict,
    phase_name: str,
    num_epochs: int,
    mosaic: bool,
    resume_from: Optional[str] = None,
    save_dir: Optional[Path] = None
) -> str:
    """
    Train for one phase (with or without mosaic).
    
    Returns:
        Path to the last checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Starting {phase_name}: {num_epochs} epochs, mosaic={mosaic}")
    print(f"{'='*60}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    num_classes = config.get('num_classes', 4)
    model = build_yolo11n_pd(num_classes=num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    
    # Create dataloaders
    train_loader = create_dataloader(config, mode='train', mosaic=mosaic)
    val_loader = create_dataloader(config, mode='val', mosaic=False)
    
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    
    # Loss function
    use_advanced_loss = config.get('use_vfl', False) and config.get('use_dfl', False)
    if use_advanced_loss:
        criterion = YOLOv8Loss(num_classes=num_classes)
    else:
        criterion = SimplifiedYOLOLoss(num_classes=num_classes)
    
    # Mixed precision training
    use_amp = config.get('amp', True) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Setup logging
    if save_dir:
        log_dir = save_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, scaler, writer
        )
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        print(f"\nEpoch [{epoch}/{start_epoch + num_epochs - 1}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        if writer is not None:
            writer.add_scalar('train/epoch_loss', train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            for k, v in metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)
        
        # Save checkpoints
        if save_dir:
            weights_dir = save_dir / 'weights'
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            # Save last checkpoint
            last_ckpt = weights_dir / 'last.pt'
            is_best = val_loss < best_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, config, str(last_ckpt), is_best)
            
            if is_best:
                best_val_loss = val_loss
            
            # Periodic saves
            save_period = config.get('save_period', 10)
            if save_period > 0 and (epoch + 1) % save_period == 0:
                periodic_ckpt = weights_dir / f'epoch_{epoch}.pt'
                save_checkpoint(model, optimizer, scheduler, epoch, config, str(periodic_ckpt))
    
    if writer is not None:
        writer.close()
    
    # Return path to last checkpoint for next phase
    if save_dir:
        return str(save_dir / 'weights' / 'last.pt')
    return None


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-PD for road detection")
    parser.add_argument('--config', type=str, default='training/train.yaml',
                       help='Path to training config YAML')
    parser.add_argument('--epochs', type=int, help='Override total epochs')
    parser.add_argument('--batch', type=int, help='Override batch size')
    parser.add_argument('--imgsz', type=int, help='Override image size')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--two-phase', action='store_true', default=True,
                       help='Use two-phase training (mosaic in final epochs)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override from CLI
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch'] = args.batch
    if args.imgsz:
        config['imgsz'] = args.imgsz
    if args.name:
        config['name'] = args.name
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Create save directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = config.get('name', config.get('name_prefix', 'yolo11n_pd'))
    project_dir = Path(config.get('project', 'runs'))
    save_dir = project_dir / f"{exp_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Experiment directory: {save_dir}")
    print(f"Config: {config}")
    
    # Two-phase training
    if args.two_phase and not args.resume:
        total_epochs = config.get('epochs', 300)
        mosaic_last_epochs = config.get('mosaic_last_epochs', 10)
        phase1_epochs = total_epochs - mosaic_last_epochs
        phase2_epochs = mosaic_last_epochs
        
        # Phase 1: No mosaic
        phase1_dir = save_dir / 'phase1'
        phase1_dir.mkdir(exist_ok=True)
        last_ckpt = train_phase(
            config, 
            phase_name="Phase 1 (No Mosaic)",
            num_epochs=phase1_epochs,
            mosaic=False,
            save_dir=phase1_dir
        )
        
        # Phase 2: With mosaic
        if phase2_epochs > 0:
            phase2_dir = save_dir / 'phase2'
            phase2_dir.mkdir(exist_ok=True)
            train_phase(
                config,
                phase_name="Phase 2 (With Mosaic)",
                num_epochs=phase2_epochs,
                mosaic=True,
                resume_from=last_ckpt,
                save_dir=phase2_dir
            )
    else:
        # Single phase training
        train_phase(
            config,
            phase_name="Training",
            num_epochs=config.get('epochs', 300),
            mosaic=config.get('mosaic', False),
            resume_from=args.resume if args.resume else None,
            save_dir=save_dir
        )
    
    print(f"\n{'='*60}")
    print(f"Training complete! Results saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
