"""
dataset.py

Dataset loader for road damage detection (RDD2022 format).

Supports:
- YOLO format annotations (class x_center y_center width height)
- Image augmentations (mosaic, mixup, HSV, flip, etc.)
- Multi-scale training
- Efficient caching
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoadDamageDataset(Dataset):
    """
    Road damage detection dataset loader.
    
    Expected directory structure:
        data/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    
    Label format (YOLO): class x_center y_center width height (normalized 0-1)
    """
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        augment: bool = True,
        mosaic: bool = False,
        mosaic_prob: float = 0.5,
        cache_images: bool = False
    ):
        """
        Args:
            img_dir: Path to images directory
            label_dir: Path to labels directory
            img_size: Target image size
            augment: Whether to apply augmentations
            mosaic: Whether to use mosaic augmentation
            mosaic_prob: Probability of applying mosaic
            cache_images: Cache images in memory for faster training
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.cache_images = cache_images
        
        # Get image paths
        self.img_files = self._get_image_files()
        self.label_files = [self._img2label_path(x) for x in self.img_files]
        
        # Cache
        self.imgs = [None] * len(self.img_files) if cache_images else None
        self.labels = [None] * len(self.img_files)
        
        # Load labels
        self._load_labels()
        
        # Augmentation pipeline
        self.transform = self._build_transforms()
        
        print(f"Dataset initialized: {len(self.img_files)} images")
    
    def _get_image_files(self) -> List[Path]:
        """Get all image file paths."""
        img_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for fmt in img_formats:
            files.extend(self.img_dir.glob(f'*{fmt}'))
            files.extend(self.img_dir.glob(f'*{fmt.upper()}'))
        return sorted(files)
    
    def _img2label_path(self, img_path: Path) -> Path:
        """Convert image path to label path."""
        return self.label_dir / f"{img_path.stem}.txt"
    
    def _load_labels(self):
        """Load all labels into memory."""
        for i, label_file in enumerate(self.label_files):
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = [x.split() for x in f.read().strip().splitlines()]
                    labels = np.array(labels, dtype=np.float32)
                self.labels[i] = labels
            else:
                self.labels[i] = np.zeros((0, 5), dtype=np.float32)
    
    def _build_transforms(self):
        """Build augmentation pipeline."""
        if self.augment:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.Blur(blur_limit=3, p=0.1),
                A.GaussNoise(p=0.1),
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))
    
    def _load_image(self, index: int) -> np.ndarray:
        """Load image from disk or cache."""
        if self.imgs is not None and self.imgs[index] is not None:
            return self.imgs[index]
        
        img_path = self.img_files[index]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.cache_images:
            self.imgs[index] = img
        
        return img
    
    def _load_mosaic(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load 4 images and create mosaic augmentation.
        
        Returns:
            mosaic_img: Combined mosaic image
            mosaic_labels: Combined labels with adjusted coordinates
        """
        # Get 4 random indices (including current)
        indices = [index] + random.choices(range(len(self)), k=3)
        
        # Create mosaic canvas
        mosaic_size = self.img_size * 2
        mosaic_img = np.full((mosaic_size, mosaic_size, 3), 114, dtype=np.uint8)
        
        # Center point
        yc, xc = [int(random.uniform(self.img_size * 0.5, self.img_size * 1.5)) for _ in range(2)]
        
        mosaic_labels = []
        
        for i, idx in enumerate(indices):
            img = self._load_image(idx)
            h, w = img.shape[:2]
            
            # Resize to img_size
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Place in mosaic (4 quadrants)
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - self.img_size, 0), max(yc - self.img_size, 0), xc, yc
                x1b, y1b, x2b, y2b = self.img_size - (x2a - x1a), self.img_size - (y2a - y1a), self.img_size, self.img_size
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - self.img_size, 0), min(xc + self.img_size, mosaic_size), yc
                x1b, y1b, x2b, y2b = 0, self.img_size - (y2a - y1a), min(self.img_size, x2a - x1a), self.img_size
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - self.img_size, 0), yc, xc, min(mosaic_size, yc + self.img_size)
                x1b, y1b, x2b, y2b = self.img_size - (x2a - x1a), 0, self.img_size, min(y2a - y1a, self.img_size)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + self.img_size, mosaic_size), min(mosaic_size, yc + self.img_size)
                x1b, y1b, x2b, y2b = 0, 0, min(self.img_size, x2a - x1a), min(y2a - y1a, self.img_size)
            
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust labels
            labels = self.labels[idx].copy()
            if len(labels):
                # Convert from normalized to pixel coordinates
                labels[:, 1] = labels[:, 1] * self.img_size + x1a - x1b
                labels[:, 2] = labels[:, 2] * self.img_size + y1a - y1b
                labels[:, 3] = labels[:, 3] * self.img_size
                labels[:, 4] = labels[:, 4] * self.img_size
                mosaic_labels.append(labels)
        
        # Concatenate labels
        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            # Clip to mosaic bounds
            mosaic_labels[:, 1:5] = np.clip(mosaic_labels[:, 1:5], 0, mosaic_size)
            # Convert back to normalized
            mosaic_labels[:, 1] = mosaic_labels[:, 1] / mosaic_size
            mosaic_labels[:, 2] = mosaic_labels[:, 2] / mosaic_size
            mosaic_labels[:, 3] = mosaic_labels[:, 3] / mosaic_size
            mosaic_labels[:, 4] = mosaic_labels[:, 4] / mosaic_size
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)
        
        # Resize mosaic to target size
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        return mosaic_img, mosaic_labels
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item from dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict with 'boxes' and 'labels'
        """
        # Mosaic augmentation
        if self.mosaic and random.random() < self.mosaic_prob:
            img, labels = self._load_mosaic(index)
        else:
            img = self._load_image(index)
            labels = self.labels[index].copy()
        
        # Prepare for albumentations
        if len(labels):
            bboxes = labels[:, 1:5].tolist()
            class_labels = labels[:, 0].astype(int).tolist()
        else:
            bboxes = []
            class_labels = []
        
        # Apply transforms
        try:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        except Exception as e:
            # Fallback: return image without boxes
            print(f"Transform error at index {index}: {e}")
            img = self.transform(image=img, bboxes=[], class_labels=[])['image']
            bboxes = []
            class_labels = []
        
        # Convert to target format
        if len(bboxes):
            boxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        targets = {
            'boxes': boxes,
            'labels': labels
        }
        
        return img, targets


def collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets


def create_dataloader(
    img_dir: str,
    label_dir: str,
    batch_size: int = 16,
    img_size: int = 640,
    augment: bool = True,
    mosaic: bool = False,
    num_workers: int = 8,
    shuffle: bool = True,
    cache_images: bool = False
):
    """
    Create dataloader for training or validation.
    
    Args:
        img_dir: Path to images directory
        label_dir: Path to labels directory
        batch_size: Batch size
        img_size: Target image size
        augment: Whether to apply augmentations
        mosaic: Whether to use mosaic augmentation
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle data
        cache_images: Cache images in memory
    
    Returns:
        DataLoader instance
    """
    dataset = RoadDamageDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        augment=augment,
        mosaic=mosaic,
        cache_images=cache_images
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True if shuffle else False
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset
    dataset = RoadDamageDataset(
        img_dir='data/images/train',
        label_dir='data/labels/train',
        img_size=640,
        augment=True,
        mosaic=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    img, targets = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Boxes shape: {targets['boxes'].shape}")
    print(f"Labels shape: {targets['labels'].shape}")
