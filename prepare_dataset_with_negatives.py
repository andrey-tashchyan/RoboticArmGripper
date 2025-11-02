#!/usr/bin/env python3
"""
Prepare YOLOv8 dataset with negative background images for false positive reduction.
Copies positive samples with labels and negative samples without labels.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Source directories - Auto-detect (Windows/macOS/Linux compatible)
SCRIPT_DIR = Path(__file__).parent.resolve()
RASPBERRY_DIR = SCRIPT_DIR / "Raspberry"
BACKGROUND_DIR = SCRIPT_DIR / "background"
TARGET_DIR = SCRIPT_DIR / "data" / "raspberries"

def copy_positive_samples(split='train'):
    """Copy positive raspberry images and their labels."""
    src_images = RASPBERRY_DIR / split / "images"
    src_labels = RASPBERRY_DIR / split / "labels"
    dst_images = TARGET_DIR / "images" / split
    dst_labels = TARGET_DIR / "labels" / split

    if not src_images.exists():
        print(f"Warning: {src_images} does not exist, skipping {split} positive samples")
        return 0

    # Get all image files
    image_files = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png"))

    copied = 0
    print(f"Copying {len(image_files)} {split} positive samples...")
    for i, img_file in enumerate(image_files, 1):
        # Copy image
        shutil.copy2(img_file, dst_images / img_file.name)

        # Copy corresponding label if it exists
        label_file = src_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, dst_labels / label_file.name)
        else:
            # Create empty label file for images without detections
            (dst_labels / f"{img_file.stem}.txt").touch()

        copied += 1
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(image_files)}")

    return copied

def copy_negative_samples(split='train'):
    """Copy negative background images WITHOUT labels."""
    src_images = BACKGROUND_DIR / split / "images"
    dst_images = TARGET_DIR / "images" / split
    # Note: We intentionally do NOT copy to labels directory

    if not src_images.exists():
        print(f"Warning: {src_images} does not exist, skipping {split} negative samples")
        return 0

    # Get all image files
    image_files = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png"))

    copied = 0
    print(f"Copying {len(image_files)} {split} negative samples (no labels)...")
    for i, img_file in enumerate(image_files, 1):
        # Copy image only (no label file)
        dst_path = dst_images / f"neg_{img_file.name}"
        shutil.copy2(img_file, dst_path)
        copied += 1
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(image_files)}")

    return copied

def verify_dataset():
    """Verify the dataset structure and count negatives."""
    for split in ['train', 'val', 'test']:
        images_dir = TARGET_DIR / "images" / split
        labels_dir = TARGET_DIR / "labels" / split

        if not images_dir.exists():
            continue

        image_files = set([f.stem for f in images_dir.glob("*.jpg")] +
                         [f.stem for f in images_dir.glob("*.png")])
        label_files = set([f.stem for f in labels_dir.glob("*.txt")])

        images_without_labels = image_files - label_files

        print(f"\n{split.upper()} split:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Images with labels: {len(label_files)}")
        print(f"  Images without labels (negatives): {len(images_without_labels)}")

        # Check for empty label files (images with no objects)
        empty_labels = 0
        for label_file in label_files:
            label_path = labels_dir / f"{label_file}.txt"
            if label_path.stat().st_size == 0:
                empty_labels += 1

        print(f"  Images with empty labels (no objects): {empty_labels}")

def main():
    print("\n" + "=" * 80)
    print("🍓 PREPARING YOLOV8 DATASET WITH NEGATIVE SAMPLES 🍓".center(80))
    print("=" * 80)

    # Process each split (use 'valid' for validation as that's the actual folder name)
    splits = [('train', 'train'), ('valid', 'val'), ('test', 'test')]

    total_positives = 0
    total_negatives = 0

    for src_split, dst_split in splits:
        print(f"\n📂 Processing {dst_split.upper()} split (from {src_split})")
        print("-" * 80)

        # Copy positive samples (with labels)
        src_images = RASPBERRY_DIR / src_split / "images"
        src_labels = RASPBERRY_DIR / src_split / "labels"
        dst_images = TARGET_DIR / "images" / dst_split
        dst_labels = TARGET_DIR / "labels" / dst_split

        if src_images.exists():
            image_files = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png"))
            print(f"✅ Copying {len(image_files)} positive samples (with labels)...")

            for img_file in tqdm(image_files, desc="  Positives", unit="img",
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                shutil.copy2(img_file, dst_images / img_file.name)
                label_file = src_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, dst_labels / label_file.name)
                else:
                    (dst_labels / f"{img_file.stem}.txt").touch()
                total_positives += 1
        else:
            print(f"⚠️  Warning: {src_images} does not exist")

        # Copy negative samples (without labels) - only for train split
        if dst_split == 'train':
            bg_images = BACKGROUND_DIR / src_split / "images"
            if bg_images.exists():
                image_files = list(bg_images.glob("*.jpg")) + list(bg_images.glob("*.png"))
                print(f"❌ Copying {len(image_files)} negative samples (no labels)...")

                for img_file in tqdm(image_files, desc="  Negatives", unit="img",
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                    dst_path = dst_images / f"neg_{img_file.name}"
                    shutil.copy2(img_file, dst_path)
                    total_negatives += 1

    print("\n" + "=" * 80)
    print("✅ DATASET PREPARATION COMPLETE ✅".center(80))
    print("=" * 80)

    # Display summary with visual bars
    print(f"\n📊 {'Summary':^76} 📊")
    print("-" * 80)
    print(f"  ✅ Positive samples (with labels): {total_positives:>5}")
    print(f"     {'█' * min(50, total_positives // 10)}")
    print(f"  ❌ Negative samples (no labels):   {total_negatives:>5}")
    print(f"     {'█' * min(50, total_negatives // 13)}")
    print(f"  📈 Total training images:          {total_positives + total_negatives:>5}")
    print("-" * 80)

    # Verify dataset with progress
    print(f"\n🔍 {'Dataset Verification':^76}")
    print("=" * 80)
    print("Scanning directories...")
    verify_dataset()

    print("\n" + "=" * 80)
    print("💡 IMPORTANT")
    print("=" * 80)
    print("✓ Negative images without label files will be treated as true negatives")
    print("  (empty scenes) by YOLOv8 during training.")
    print("✓ This helps reduce false positives on hands, faces, and red objects.")
    print("✓ Ready to train with: python3 train_with_negatives.py --mode fast|full")
    print("=" * 80)

if __name__ == "__main__":
    main()
