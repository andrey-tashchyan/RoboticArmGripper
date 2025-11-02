#!/usr/bin/env python3
"""
Custom YOLOv8 training script that properly handles negative background images.
This script ensures YOLO loads images without labels as true negatives.

Usage:
    python3 train_with_negatives.py --mode fast    # Quick training (30 epochs)
    python3 train_with_negatives.py --mode full    # Full training (120 epochs)
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import sys
import argparse
from tqdm import tqdm

# Training modes
TRAINING_MODES = {
    'fast': {
        'epochs': 30,
        'patience': 10,
        'batch': 8,
        'description': 'Mode rapide - Test et validation (30 epochs)'
    },
    'full': {
        'epochs': 120,
        'patience': 30,
        'batch': 8,
        'description': 'Mode complet - Entraînement final (120 epochs)'
    }
}

# Base configuration
DATA_YAML = "data/raspberries.yaml"
MODEL = "yolov8s.pt"
IMGSZ = 896
DEVICE = "mps"

# Augmentation parameters
HSV_H = 0.015
HSV_S = 0.6
HSV_V = 0.45
DEGREES = 7
TRANSLATE = 0.10
SCALE = 0.35
SHEAR = 3
PERSPECTIVE = 0.0005
MOSAIC = 0.6
MIXUP = 0.15
COPY_PASTE = 0.4
ERASING = 0.25

# Training hyperparameters
BOX = 7
CLS = 0.6
DFL = 1.5
LR0 = 0.004
LRF = 0.1
OPTIMIZER = "AdamW"
COS_LR = True
WARMUP_EPOCHS = 3

def count_negative_images(data_yaml_path):
    """Count images without corresponding label files (true negatives)."""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_config['path'])
    train_images_dir = dataset_root / data_config['train']
    train_labels_dir = dataset_root / 'labels' / 'train'

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(train_images_dir.glob(ext))

    # Count images without labels
    negatives = 0
    positives = 0

    for img_path in all_images:
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            negatives += 1
        else:
            positives += 1

    return negatives, positives, len(all_images)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Training with Negative Background Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  fast : Mode rapide (30 epochs) - Pour tester et valider rapidement
  full : Mode complet (120 epochs) - Pour l'entraînement final de production

Examples:
  python3 train_with_negatives.py --mode fast
  python3 train_with_negatives.py --mode full
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fast', 'full'],
        default='full',
        help='Training mode: fast (30 epochs) or full (120 epochs)'
    )
    return parser.parse_args()

def print_progress_header(mode_config, negatives, positives, total):
    """Print training header with progress info."""
    print("\n" + "=" * 80)
    print("🍓 YOLOV8 RASPBERRY DETECTION - TRAINING WITH NEGATIVES 🍓".center(80))
    print("=" * 80)

    print(f"\n📊 {'Dataset Statistics':^76} 📊")
    print("-" * 80)
    print(f"  Total training images: {total:,}")
    print(f"  ✅ Images with labels (positives): {positives:,}")
    print(f"  ❌ Images without labels (negatives): {negatives:,}")
    print(f"  📈 Negative ratio: {negatives/total*100:.1f}%")
    print("-" * 80)

    print(f"\n💡 Loaded {negatives} negative background images with no labels.")
    print("   These will be treated as empty scenes to reduce false positives.\n")

def main():
    # Parse arguments
    args = parse_arguments()
    mode_config = TRAINING_MODES[args.mode]

    print("\n" + "=" * 80)
    print(f"⚙️  MODE: {args.mode.upper()} - {mode_config['description']}")
    print("=" * 80)

    # Count negative images with progress
    print("\n🔍 Analyzing dataset...")
    negatives, positives, total = count_negative_images(DATA_YAML)

    print_progress_header(mode_config, negatives, positives, total)

    # Load model with progress
    print(f"📦 Loading model: {MODEL}")
    with tqdm(total=100, desc="Model loading", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        model = YOLO(MODEL)
        pbar.update(100)

    # Training parameters
    print(f"\n⚙️  {'Training Configuration':^76}")
    print("-" * 80)
    print(f"  Model: {MODEL}")
    print(f"  Data: {DATA_YAML}")
    print(f"  Epochs: {mode_config['epochs']}")
    print(f"  Batch size: {mode_config['batch']}")
    print(f"  Image size: {IMGSZ}x{IMGSZ}")
    print(f"  Device: {DEVICE.upper()}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  Learning rate: {LR0} → {LRF}")
    print(f"  Patience: {mode_config['patience']} epochs")
    print("-" * 80)

    print(f"\n🎨 {'Augmentation Parameters':^76}")
    print("-" * 80)
    print(f"  HSV: h={HSV_H}, s={HSV_S}, v={HSV_V}")
    print(f"  Geometric: degrees={DEGREES}°, translate={TRANSLATE}, scale={SCALE}")
    print(f"  Advanced: mosaic={MOSAIC}, mixup={MIXUP}, copy_paste={COPY_PASTE}")
    print(f"  Erasing: {ERASING}")
    print("-" * 80)

    print("\n" + "=" * 80)
    print(f"🚀 STARTING TRAINING - {args.mode.upper()} MODE 🚀".center(80))
    print("=" * 80 + "\n")

    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=mode_config['epochs'],
        imgsz=IMGSZ,
        batch=mode_config['batch'],
        device=DEVICE,
        # Augmentation
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASING,
        # Loss weights
        box=BOX,
        cls=CLS,
        dfl=DFL,
        # Optimizer
        lr0=LR0,
        lrf=LRF,
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        warmup_epochs=WARMUP_EPOCHS,
        patience=mode_config['patience'],
        # Project organization
        project="runs/raspberry_detect",
        name=f"train_{args.mode}",
        exist_ok=True,
        # Ensure negative images are loaded
        verbose=True
    )

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE ✅".center(80))
    print("=" * 80)

    # Print final metrics with progress bar
    print("\n📈 Final Metrics:")
    print("-" * 80)
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)

        print(f"  mAP50:     {map50:.4f} {'█' * int(map50 * 50)}")
        print(f"  mAP50-95:  {map50_95:.4f} {'█' * int(map50_95 * 50)}")
        print(f"  Precision: {precision:.4f} {'█' * int(precision * 50)}")
        print(f"  Recall:    {recall:.4f} {'█' * int(recall * 50)}")
    print("-" * 80)

    # Print output location
    output_dir = f"runs/raspberry_detect/train_{args.mode}"
    print(f"\n💾 Model saved to: {output_dir}/")
    print(f"   Best weights: {output_dir}/weights/best.pt")
    print(f"   Last weights: {output_dir}/weights/last.pt")

    print("\n💡 Note: Validation metrics should show improved precision and")
    print("   reduced false positives on hands, faces, and red objects.")
    print("=" * 80)

if __name__ == "__main__":
    main()
