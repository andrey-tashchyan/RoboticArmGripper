#!/usr/bin/env python3
"""
Prepare YOLOv8 dataset with negative background images.
COMPATIBLE: Windows, macOS, Linux
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Source directories - RELATIFS (fonctionne partout)
SCRIPT_DIR = Path(__file__).parent.resolve()
RASPBERRY_DIR = SCRIPT_DIR / "Raspberry"
BACKGROUND_DIR = SCRIPT_DIR / "background"
TARGET_DIR = SCRIPT_DIR / "data" / "raspberries"

def main():
    print("\n" + "=" * 80)
    print("🍓 PREPARING YOLOV8 DATASET WITH NEGATIVE SAMPLES 🍓".center(80))
    print("=" * 80)

    print(f"\nDirectories:")
    print(f"  Script:     {SCRIPT_DIR}")
    print(f"  Raspberry:  {RASPBERRY_DIR}")
    print(f"  Background: {BACKGROUND_DIR}")
    print(f"  Target:     {TARGET_DIR}")

    # Verify source directories exist
    if not RASPBERRY_DIR.exists():
        print(f"\n❌ ERROR: Raspberry directory not found: {RASPBERRY_DIR}")
        print("   Please ensure you're running this script from the project root.")
        return

    if not BACKGROUND_DIR.exists():
        print(f"\n❌ ERROR: Background directory not found: {BACKGROUND_DIR}")
        return

    # Create target directories
    for split in ['train', 'val', 'test']:
        (TARGET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each split
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

    # Display summary
    print(f"\n📊 {'Summary':^76} 📊")
    print("-" * 80)
    print(f"  ✅ Positive samples (with labels): {total_positives:>5}")
    print(f"  ❌ Negative samples (no labels):   {total_negatives:>5}")
    print(f"  📈 Total training images:          {total_positives + total_negatives:>5}")
    print("-" * 80)

    print("\n💡 Ready to train with:")
    print("   python train_with_negatives.py --mode fast")
    print("   python train_with_negatives.py --mode full")
    print("=" * 80)

if __name__ == "__main__":
    main()
