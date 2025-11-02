# YOLOv8 Training with Negative Background Images - Setup Summary

## Overview
Successfully configured YOLOv8 training pipeline to incorporate 669 negative background images as true negatives to reduce false positives on hands, faces, and red objects.

## Dataset Structure

### Location
- **Dataset Root**: `data/raspberries/`
- **Configuration**: [data/raspberries.yaml](data/raspberries.yaml)

### Dataset Statistics
```
Total training images: 1,168
├── Positive samples (with labels): 499 (42.7%)
└── Negative samples (no labels): 669 (57.3%)

Validation images: 99
Test images: 51
```

### Directory Structure
```
data/raspberries/
├── images/
│   ├── train/          # 1,168 images (499 positives + 669 negatives)
│   ├── val/            # 99 validation images
│   └── test/           # 51 test images
└── labels/
    ├── train/          # 499 label files (only for positives)
    ├── val/            # 99 label files
    └── test/           # 51 label files
```

### Key Features
- ✅ **Single detection class**: `raspberry` (nc=1)
- ✅ **Negative images**: 669 images WITHOUT corresponding .txt label files
- ✅ **YOLO interpretation**: Images without labels are treated as empty scenes (true negatives)
- ✅ **Naming convention**: Negative images prefixed with `neg_` for easy identification

## Training Configuration

### Model & Hardware
- **Model**: YOLOv8s (yolov8s.pt)
- **Device**: MPS (Apple Silicon)
- **Image size**: 896x896
- **Epochs**: 120
- **Patience**: 30 epochs (early stopping)

### Optimizer
- **Type**: AdamW
- **Initial LR**: 0.004
- **Final LR**: 0.1
- **Cosine LR**: True
- **Warmup epochs**: 3

### Augmentation Parameters
- **HSV**: h=0.015, s=0.6, v=0.45
- **Geometric**:
  - Degrees: 7
  - Translate: 0.10
  - Scale: 0.35
  - Shear: 3
  - Perspective: 0.0005
- **Advanced**:
  - Mosaic: 0.6
  - Mixup: 0.15
  - Copy-paste: 0.4
  - Erasing: 0.25

### Loss Weights
- **Box**: 7
- **Class**: 0.6
- **DFL**: 1.5

## Scripts

### 1. Dataset Preparation: [prepare_dataset_with_negatives.py](prepare_dataset_with_negatives.py)
**Purpose**: Copies positive and negative images to the training dataset

**Features**:
- Copies 499 positive raspberry images with their label files
- Copies 669 negative background images WITHOUT label files
- Verifies dataset structure and counts
- Ensures YOLO treats unlabeled images as true negatives

**Usage**:
```bash
python3 prepare_dataset_with_negatives.py
```

**Output**:
```
Loaded 669 negative background images with no labels.
train: Scanning... 499 images, 669 backgrounds, 0 corrupt
```

### 2. Training Script: [train_with_negatives.py](train_with_negatives.py)
**Purpose**: Custom YOLOv8 training with negative image counting and verification

**Features**:
- Displays dataset statistics before training
- Shows "Loaded N negative background images with no labels" message
- Applies all specified training parameters
- Saves results to `runs/raspberry_detect/train_with_negatives/`

**Usage**:
```bash
source .venv311/bin/activate
python3 train_with_negatives.py
```

## Training Command (Alternative CLI Method)

You can also train directly using the YOLO CLI:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=data/raspberries.yaml \
  epochs=120 \
  imgsz=896 \
  device=mps \
  hsv_h=0.015 \
  hsv_s=0.6 \
  hsv_v=0.45 \
  degrees=7 \
  translate=0.10 \
  scale=0.35 \
  shear=3 \
  perspective=0.0005 \
  mosaic=0.6 \
  mixup=0.15 \
  copy_paste=0.4 \
  erasing=0.25 \
  box=7 \
  cls=0.6 \
  dfl=1.5 \
  lr0=0.004 \
  lrf=0.1 \
  optimizer=AdamW \
  cos_lr=True \
  warmup_epochs=3 \
  patience=30
```

## How Negative Images Work in YOLO

### Mechanism
1. **Image Discovery**: YOLO scans the `images/train/` directory and finds all .jpg/.png files
2. **Label Lookup**: For each image, YOLO looks for a corresponding .txt file in `labels/train/`
3. **Negative Detection**:
   - **If label file exists**: Image treated as having objects (or empty if file is empty)
   - **If NO label file**: Image treated as a TRUE NEGATIVE (empty scene)
4. **Training Impact**: Model learns that these scenes should produce zero detections

### Benefits
- ✅ Reduces false positives on similar-looking objects (hands, faces, red objects)
- ✅ Improves precision without sacrificing recall
- ✅ Model learns to distinguish raspberries from confusing backgrounds
- ✅ 57.3% negative ratio provides strong negative signal

## Verification

### Dataset Verification
The training log confirms proper loading:
```
train: Scanning .../labels/train... 499 images, 669 backgrounds, 0 corrupt
```

This shows:
- 499 images WITH labels (positives)
- 669 backgrounds WITHOUT labels (negatives)
- 0 corrupt images

### Expected Outcomes
After training, you should observe:
1. **Higher Precision**: Fewer false positives on validation/test sets
2. **Reduced False Positives**: Especially on:
   - Human hands
   - Human faces
   - Red-colored objects
   - Similar textured surfaces
3. **Maintained Recall**: Still detects actual raspberries effectively

## Training Progress

Training is currently running in the background. Check progress:

```bash
# View real-time training logs
tail -f runs/raspberry_detect/train_with_negatives/train.log

# View results after training
ls runs/raspberry_detect/train_with_negatives/
# Contains: weights/, results.png, confusion_matrix.png, etc.
```

## Model Outputs

After training completes, you'll find:

```
runs/raspberry_detect/train_with_negatives/
├── weights/
│   ├── best.pt           # Best model weights (highest mAP)
│   └── last.pt           # Final epoch weights
├── results.png           # Training curves (loss, mAP, precision, recall)
├── confusion_matrix.png  # Confusion matrix
├── labels.jpg            # Label distribution visualization
└── args.yaml            # All training arguments
```

## Next Steps

1. **Monitor Training**: Wait for training to complete (~120 epochs or early stop)
2. **Evaluate Results**: Compare precision/recall with previous training runs
3. **Test on False Positives**: Run inference on images of hands, faces, red objects
4. **Validate Performance**: Ensure false positives are significantly reduced

## Notes

- The negative images are sourced from `background/train/images/` (669 images)
- All negative images are prefixed with `neg_` in the training set
- No label files are created for negative images (intentional)
- YOLO's dataloader automatically handles missing labels as negatives
- The configuration ensures single-class detection (raspberry only)

## Contact & Issues

If you encounter issues:
1. Check that all 669 negative images are in `data/raspberries/images/train/`
2. Verify only 499 label files exist in `data/raspberries/labels/train/`
3. Ensure the YOLO training log shows "669 backgrounds"
4. Monitor validation precision increases over epochs

---

**Generated**: November 2, 2025
**Training Status**: ✅ Running (120 epochs, patience=30)
**Negative Images**: ✅ 669 loaded successfully
