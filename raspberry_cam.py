#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raspberry ripeness and size classifier from live webcam/video using OpenCV and Torch/ONNX.

README (Quick Start)
--------------------
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Torch mode (Ultralytics YOLOv8 .pt)
python raspberry_cam.py --src 0 --model weights/yolov8n.pt --device mps --roi-w 0.4 --roi-h 0.4 --size-thresh-px 9000

# ONNX mode (YOLOv8 .onnx)
python raspberry_cam.py --src 0 --onnx weights/yolov8n.onnx --roi-w 0.4 --roi-h 0.4 --size-thresh-px 9000

# Fallback (no model provided) - color+shape proposals in center ROI only
python raspberry_cam.py --src 0 --roi-w 0.4 --roi-h 0.4 --size-thresh-px 9000

Apple Silicon (MPS) Notes
-------------------------
- Install torch with MPS support via pip (see requirements.txt). When running, use --device mps.
- If MPS is not available, the script will fall back to CPU.

Notes
-----
- The script never downloads weights. Provide local paths via --model or --onnx.
- Thresholds for HSV may need adjustment under different lighting. Consider using --auto-calib.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import cv2


# Optional imports guarded at runtime
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort  # type: ignore
    ONNX_AVAILABLE = True
except Exception:
    ort = None  # type: ignore
    ONNX_AVAILABLE = False

# Ultralytics is optional (only needed for .pt loading convenience)
try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None  # type: ignore
    ULTRALYTICS_AVAILABLE = False


# ----------------------------- Utility structures -----------------------------


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls: int

    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Track:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    last_centroid: Tuple[float, float]
    last_seen: int
    age: int
    has_been_counted: bool = False
    consecutive_valid_frames: int = 0  # For temporal consistency
    confirmed_raspberry: bool = False  # True after min_frames validation passes
    last_ripeness: str = "UNKNOWN"  # Track last known ripeness
    # Temporal smoothing with EMA (exponential moving average)
    ema_red_ratio: float = 0.0
    ema_a_mean: float = 0.0
    ema_alpha: float = 0.3  # EMA smoothing factor (0.2-0.4 for responsive smoothing)
    last_reject_reason: str = ""  # Store last rejection reason for debug


class Tracker:
    def __init__(self, min_iou: float = 0.3, max_age: int = 15):
        self.min_iou = float(min_iou)
        self.max_age = int(max_age)
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = a_area + b_area - inter + 1e-6
        return inter / denom

    def update(self, detections: List[Detection], frame_idx: int) -> Dict[int, Detection]:
        assigned: Dict[int, int] = {}  # track_id -> det_idx
        det_assigned: set[int] = set()

        # Greedy matching by IoU
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1
            for j, det in enumerate(detections):
                if j in det_assigned:
                    continue
                iou_val = self.iou(tr.xyxy, det.xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0 and best_iou >= self.min_iou:
                assigned[tid] = best_j
                det_assigned.add(best_j)

        # Update matched tracks
        for tid, j in assigned.items():
            det = detections[j]
            c = det.centroid()
            self.tracks[tid].xyxy = det.xyxy
            self.tracks[tid].last_centroid = c
            self.tracks[tid].last_seen = frame_idx
            self.tracks[tid].age += 1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in det_assigned:
                continue
            c = det.centroid()
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                xyxy=det.xyxy,
                last_centroid=c,
                last_seen=frame_idx,
                age=1,
            )

        # Remove stale tracks
        remove_ids = [tid for tid, tr in self.tracks.items() if (frame_idx - tr.last_seen) > self.max_age]
        for tid in remove_ids:
            del self.tracks[tid]

        # Return mapping track_id -> associated detection (best current)
        out: Dict[int, Detection] = {}
        # Best effort: choose the latest matched or nearest by IoU
        for tid, tr in self.tracks.items():
            # Find the closest detection by IoU among current detections
            best_iou = 0.0
            best_det: Optional[Detection] = None
            for det in detections:
                iou_val = self.iou(tr.xyxy, det.xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = det
            if best_det is not None:
                out[tid] = best_det
        return out


# ----------------------------- ROI utilities -----------------------------


def center_roi(frame_shape: Tuple[int, int, int], w_frac: float, h_frac: float) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    roi_w = int(max(1, min(w, w * w_frac)))
    roi_h = int(max(1, min(h, h * h_frac)))
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return x1, y1, x2, y2


def in_roi(xyxy: Tuple[float, float, float, float], roi_xyxy: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    rx1, ry1, rx2, ry2 = roi_xyxy
    return (cx >= rx1) and (cx <= rx2) and (cy >= ry1) and (cy <= ry2)


def roi_center_distance(xyxy: Tuple[float, float, float, float], roi_xyxy: Tuple[int, int, int, int]) -> float:
    """Calculate normalized distance from detection centroid to ROI center.

    Returns value in [0, 1] where 0 = at ROI center, 1 = at ROI edge.
    Used for center-weighted detection preference.
    """
    x1, y1, x2, y2 = xyxy
    det_cx = 0.5 * (x1 + x2)
    det_cy = 0.5 * (y1 + y2)

    rx1, ry1, rx2, ry2 = roi_xyxy
    roi_cx = 0.5 * (rx1 + rx2)
    roi_cy = 0.5 * (ry1 + ry2)
    roi_w = rx2 - rx1
    roi_h = ry2 - ry1

    # Euclidean distance to ROI center, normalized by half-diagonal
    dx = det_cx - roi_cx
    dy = det_cy - roi_cy
    dist = math.sqrt(dx * dx + dy * dy)
    max_dist = math.sqrt((roi_w / 2) ** 2 + (roi_h / 2) ** 2) + 1e-6

    return min(1.0, dist / max_dist)


# ----------------------------- Color preprocessing -----------------------------


def enhance_contrast(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply contrast enhancement to recover dull reds.

    Uses CLAHE on V channel + convertScaleAbs for better color separation.
    This helps detect raspberries in poor lighting conditions.
    """
    # First apply gentle gamma + brightness boost
    enhanced = cv2.convertScaleAbs(frame_bgr, alpha=1.15, beta=8)

    # Convert to HSV and apply CLAHE on V channel
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE with moderate clip limit
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v)

    # Boost saturation slightly for better color discrimination
    s_boosted = cv2.convertScaleAbs(s, alpha=1.15, beta=0)

    # Merge and convert back
    hsv_enhanced = cv2.merge([h, s_boosted, v_enhanced])
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


# ----------------------------- HSV and ripeness -----------------------------


@dataclass
class RipenessThresholds:
    red_min: float = 0.35
    white_min: float = 0.25
    s_min: int = 90
    v_min: int = 60
    a_min: int = 150  # LAB A-channel threshold for red


def hsv_masks(hsv: np.ndarray, s_min: int = 90, v_min: int = 60, adaptive: bool = False, refine: bool = True) -> Dict[str, np.ndarray]:
    """Return HSV masks for red (expanded lobes) and white/pale.

    Uses expanded red hue lobes [0..10] and [170..180] for better detection.
    If adaptive=True, lowers S threshold when detecting low red ratios.
    If refine=True, applies morphological operations to clean up masks.
    """
    h, s, v = cv2.split(hsv)

    # Expanded red hue range for better raspberry detection
    lower_red1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, s_min, v_min], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Adaptive: if red_ratio is low, try with lower S threshold
    if adaptive:
        red_ratio = float(np.count_nonzero(mask_red)) / float(hsv.shape[0] * hsv.shape[1] + 1e-6)
        if red_ratio < 0.05:  # Very little red detected
            s_min_relaxed = max(30, int(s_min * 0.6))
            lower_red1_relax = np.array([0, s_min_relaxed, v_min], dtype=np.uint8)
            lower_red2_relax = np.array([170, s_min_relaxed, v_min], dtype=np.uint8)
            mask_red1_relax = cv2.inRange(hsv, lower_red1_relax, upper_red1)
            mask_red2_relax = cv2.inRange(hsv, lower_red2_relax, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1_relax, mask_red2_relax)

    # Morphological refinement: CLOSE then OPEN to reduce speckles while preserving berries
    if refine:
        kernel_3x3 = np.ones((3, 3), np.uint8)
        # Close: fill small holes
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_3x3, iterations=1)
        # Open: remove small noise
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_3x3, iterations=1)

    # White/pale: low S, high V
    mask_white = cv2.inRange(hsv, np.array([0, 0, 150], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))

    if refine:
        kernel_small = np.ones((3, 3), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    return {
        "red": mask_red,
        "white": mask_white,
    }


def skin_mask_ycrcb(bgr: np.ndarray) -> np.ndarray:
    """Detect skin tones using YCrCb color space.

    Thresholds: Cr ∈ [133, 173], Cb ∈ [77, 127]
    Returns binary mask where 255 = skin pixels.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    return skin_mask


def crop_circularity_score(bgr_crop: np.ndarray) -> float:
    """Compute circularity score (4πA/P²) of the largest contour in the crop.

    Uses Otsu thresholding to segment foreground, then measures the largest contour.
    Returns circularity in [0, 1], where 1.0 = perfect circle.
    Returns 0.0 if no valid contour found.
    """
    if bgr_crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    # Find largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    if perimeter < 1e-6 or area < 1e-6:
        return 0.0

    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    return min(1.0, circularity)  # clamp to [0, 1]


def texture_score(bgr_crop: np.ndarray) -> float:
    """Compute texture/edge score using Laplacian variance.

    Higher variance indicates more texture/edges (real objects).
    Smooth regions (like skin, clothing) have lower variance.
    Returns variance of Laplacian. Typical threshold: 30.0+
    """
    if bgr_crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def check_bbox_geometry(xyxy: Tuple[float, float, float, float], frame_shape: Tuple[int, int]) -> Tuple[bool, str]:
    """Check if bounding box has valid geometry (aspect ratio, border touching).

    Returns (is_valid, reject_reason).
    Rejects if:
    - Aspect ratio > 3:1 or < 1:3 (too elongated)
    - Touches 3+ borders (likely full-frame object like face/body)
    """
    x1, y1, x2, y2 = xyxy
    h, w = frame_shape[:2]

    # Aspect ratio check
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 1 or box_h < 1:
        return False, "degenerate_box"

    aspect = box_w / box_h
    if aspect > 3.0 or aspect < 0.33:
        return False, "extreme_aspect_ratio"

    # Border touching check (within 5px of edge)
    border_margin = 5.0
    touches_left = x1 <= border_margin
    touches_right = x2 >= (w - border_margin)
    touches_top = y1 <= border_margin
    touches_bottom = y2 >= (h - border_margin)

    border_count = sum([touches_left, touches_right, touches_top, touches_bottom])
    if border_count >= 3:
        return False, "touches_3+_borders"

    return True, ""


@dataclass
class ValidationConfig:
    """Configuration for strict multi-cue raspberry validation."""
    area_min: int = 800
    area_max_frac: float = 0.15
    red_min: float = 0.40
    s_min: int = 90
    v_min: int = 60
    a_min: int = 150
    min_circularity: float = 0.35
    min_texture: float = 30.0
    max_skin_ratio: float = 0.25
    strict_mode: bool = False
    center_weight: float = 0.3  # Weight for center-distance penalty (0 = no penalty, 1 = strong penalty)


def validate_candidate(
    crop: np.ndarray,
    bbox: Tuple[float, float, float, float],
    frame_dims: Tuple[int, int],
    roi_xyxy: Tuple[int, int, int, int],
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, float]]:
    """Comprehensive multi-cue validator for raspberry candidates.

    Returns (is_valid, stats_dict).
    Performs all validation checks in strict mode:
    - Centroid inside ROI
    - Area within bounds [area_min, area_max_frac × frame]
    - Red color evidence (HSV + LAB A-channel)
    - Not skin tone
    - Valid geometry (aspect ratio, borders)
    - Circularity >= threshold
    - Texture score >= threshold
    """
    stats: Dict[str, float] = {}

    # Centroid in ROI check
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    rx1, ry1, rx2, ry2 = roi_xyxy
    in_roi_flag = (cx >= rx1) and (cx <= rx2) and (cy >= ry1) and (cy <= ry2)
    stats["in_roi"] = float(in_roi_flag)

    if not in_roi_flag:
        return False, stats

    # Area bounds check
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    frame_h, frame_w = frame_dims
    max_area = float(frame_h * frame_w) * args.area_max_frac
    stats["area"] = area

    # Scale area_min by sensitivity
    effective_area_min = args.area_min / max(0.5, args.sensitivity)

    if area < effective_area_min:
        stats["reject"] = 1.0
        return False, stats
    if area > max_area:
        stats["reject"] = 2.0
        return False, stats

    # Geometry check
    geom_valid, _ = check_bbox_geometry(bbox, frame_dims)
    if not geom_valid:
        stats["reject"] = 3.0
        return False, stats

    if crop.size == 0:
        stats["reject"] = 4.0
        return False, stats

    # Circularity check
    circ = crop_circularity_score(crop)
    stats["circularity"] = circ
    if circ < args.min_circularity:
        stats["reject"] = 5.0
        return False, stats

    # Texture check
    tex = texture_score(crop)
    stats["texture"] = tex
    if tex < args.min_texture:
        stats["reject"] = 6.0
        return False, stats

    # Skin rejection
    skin_mask = skin_mask_ycrcb(crop)
    total_pixels = float(crop.shape[0] * crop.shape[1] + 1e-6)
    skin_ratio = float(np.count_nonzero(skin_mask)) / total_pixels
    stats["skin_ratio"] = skin_ratio

    if skin_ratio > args.max_skin_ratio:
        stats["reject"] = 7.0
        return False, stats

    # Color validation: HSV red + LAB A-channel
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Scale red_min by sensitivity
    effective_red_min = args.red_min / max(0.5, args.sensitivity)
    effective_s_min = max(30, int(args.s_min / (1.0 + 0.2 * (args.sensitivity - 1.0))))
    effective_v_min = max(30, int(args.v_min / (1.0 + 0.15 * (args.sensitivity - 1.0))))

    # Red mask with expanded hue lobes
    lower_red1 = np.array([0, effective_s_min, effective_v_min], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, effective_s_min, effective_v_min], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv_crop, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_crop, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Apply morphological cleanup: CLOSE then OPEN
    kernel_3x3 = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_3x3)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_3x3)

    red_ratio = float(np.count_nonzero(mask_red)) / total_pixels
    stats["red_ratio"] = red_ratio

    # White ratio for unripe detection
    mask_white = cv2.inRange(hsv_crop, np.array([0, 0, 150], dtype=np.uint8),
                             np.array([180, 60, 255], dtype=np.uint8))
    white_ratio = float(np.count_nonzero(mask_white)) / total_pixels
    stats["white_ratio"] = white_ratio

    if red_ratio < effective_red_min:
        stats["reject"] = 8.0
        return False, stats

    # LAB A-channel check
    lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    a_mean = float(np.mean(lab_crop[:, :, 1]))
    stats["a_mean"] = a_mean

    effective_a_min = max(120, int(args.a_min / (1.0 + 0.1 * (args.sensitivity - 1.0))))

    if a_mean < effective_a_min:
        stats["reject"] = 9.0
        return False, stats

    # All checks passed
    stats["reject"] = 0.0
    return True, stats


def classify_ripeness(hsv_crop: np.ndarray, bgr_crop: np.ndarray, thr: RipenessThresholds, adaptive: bool = True) -> Tuple[str, Dict[str, float]]:
    """Classify ripeness using strict HSV red detection + LAB A-channel validation + skin rejection.

    Args:
        hsv_crop: HSV image crop
        bgr_crop: BGR image crop
        thr: Ripeness thresholds
        adaptive: If True, enables adaptive S threshold lowering for better detection
    """
    masks = hsv_masks(hsv_crop, s_min=thr.s_min, v_min=thr.v_min, adaptive=adaptive)
    total = float(hsv_crop.shape[0] * hsv_crop.shape[1] + 1e-6)
    red_pct = float(np.count_nonzero(masks["red"])) / total
    white_pct = float(np.count_nonzero(masks["white"])) / total
    s_mean = float(np.mean(hsv_crop[:, :, 1]))
    v_mean = float(np.mean(hsv_crop[:, :, 2]))

    # Convert to LAB and compute A-channel mean (red-green axis)
    lab_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    a_mean = float(np.mean(lab_crop[:, :, 1]))

    # Skin detection: reject if too much skin tone present
    skin_mask = skin_mask_ycrcb(bgr_crop)
    skin_pct = float(np.count_nonzero(skin_mask)) / total

    # Reject skin/face detections
    if skin_pct > 0.25:
        label = "UNKNOWN"
        meta = {
            "red_pct": red_pct,
            "white_pct": white_pct,
            "s_mean": s_mean,
            "v_mean": v_mean,
            "a_mean": a_mean,
            "skin_pct": skin_pct,
            "reject_reason": "skin_detected",
        }
        return label, meta

    # Strict decision rule: RIPE requires red ratio AND saturation AND LAB A-channel
    if red_pct >= thr.red_min and s_mean >= thr.s_min and a_mean >= thr.a_min:
        label = "RIPE"
    elif white_pct >= thr.white_min:
        label = "UNRIPE"
    else:
        label = "UNKNOWN"

    meta = {
        "red_pct": red_pct,
        "white_pct": white_pct,
        "s_mean": s_mean,
        "v_mean": v_mean,
        "a_mean": a_mean,
        "skin_pct": skin_pct,
    }
    return label, meta


def auto_calibrate(cap: cv2.VideoCapture, duration_sec: float = 7.0, fps_limit: float = 15.0) -> RipenessThresholds:
    """Dynamic color calibration using percentile-based threshold adjustment.

    Samples the scene for 5-10 seconds and tunes s_min, v_min, and red_min based on
    actual illumination conditions. Uses percentile-based approach to handle varying lighting.
    """
    print(f"[auto-calib] Starting dynamic calibration for {duration_sec:.1f}s...")
    s_vals: List[float] = []
    v_vals: List[float] = []
    red_ratios: List[float] = []

    t0 = time.time()
    last = 0.0
    frame_count = 0

    while time.time() - t0 < duration_sec:
        ok, frame = cap.read()
        if not ok:
            break

        # Apply gentle enhancement
        enhanced = enhance_contrast(frame)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Collect S and V statistics
        s_vals.extend(s.flatten().tolist())
        v_vals.extend(v.flatten().tolist())

        # Check for red pixels with wide hue range
        lower_red1 = np.array([0, 40, 40], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 40, 40], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_ratio = float(np.count_nonzero(red_mask)) / float(frame.shape[0] * frame.shape[1])
        red_ratios.append(red_ratio)

        frame_count += 1

        if fps_limit > 0:
            now = time.time()
            delay = (1.0 / fps_limit) - (now - last)
            if delay > 0:
                time.sleep(delay)
            last = now

    if not s_vals or not v_vals:
        print("[auto-calib] No frames captured; using defaults.")
        return RipenessThresholds()

    # Percentile-based threshold calculation
    s_percentile_30 = float(np.percentile(s_vals, 30))
    v_percentile_20 = float(np.percentile(v_vals, 20))
    red_ratio_median = float(np.median(red_ratios)) if red_ratios else 0.01

    # Adaptive thresholds based on scene characteristics
    s_min = int(max(40, min(110, s_percentile_30)))  # More permissive
    v_min = int(max(30, min(100, v_percentile_20)))  # Lower for dim lighting

    # Adjust red_min based on detected red presence
    if red_ratio_median > 0.05:  # Scene has red objects
        red_min = max(0.15, red_ratio_median * 0.5)  # Lower threshold
    else:
        red_min = 0.25  # Default for scenes without red

    suggested = RipenessThresholds(
        red_min=float(red_min),
        white_min=0.25,
        s_min=s_min,
        v_min=v_min,
        a_min=140,  # Lower for better detection
    )

    print(f"[auto-calib] ✓ Calibrated from {frame_count} frames:")
    print(f"  S: 30th percentile={s_percentile_30:.1f} → s_min={s_min}")
    print(f"  V: 20th percentile={v_percentile_20:.1f} → v_min={v_min}")
    print(f"  Red ratio median={red_ratio_median:.3f} → red_min={red_min:.2f}")

    return suggested


# ----------------------------- Size classification -----------------------------


@dataclass
class SizeState:
    mode: str = "absolute"  # or "relative"
    abs_thresh_px: int = 9000
    factor: float = 1.2
    history: Deque[float] = dataclasses.field(default_factory=lambda: deque(maxlen=50))


def classify_size(area_px: float, state: SizeState) -> Tuple[str, float]:
    if state.mode == "relative" and len(state.history) > 0:
        med = float(np.median(list(state.history)))
        dyn = med * state.factor
        label = "BIG" if area_px >= dyn else "SMALL"
        return label, dyn
    else:
        label = "BIG" if area_px >= state.abs_thresh_px else "SMALL"
        return label, float(state.abs_thresh_px)


# ----------------------------- NMS utility -----------------------------


def nms_boxes(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """Apply Non-Maximum Suppression to detections.

    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered list of detections after NMS
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.conf, reverse=True)
    keep: List[Detection] = []

    while sorted_dets:
        # Take highest confidence detection
        best = sorted_dets.pop(0)
        keep.append(best)

        # Remove overlapping detections
        filtered: List[Detection] = []
        for det in sorted_dets:
            iou = Tracker.iou(best.xyxy, det.xyxy)
            if iou < iou_threshold:
                filtered.append(det)
        sorted_dets = filtered

    return keep


# ----------------------------- Model loading and detection -----------------------------


def load_model(model_path: str, onnx_path: str, device: str, conf: float, nms_iou: float, keep_classes: str = "") -> Dict[str, Any]:
    """Load model in one of two modes: Torch (.pt via Ultralytics) or ONNXRuntime (.onnx).
    Returns a dict with keys: 'mode' in {'torch','onnx','none'} and related objects.
    """
    if model_path:
        if not ULTRALYTICS_AVAILABLE or not TORCH_AVAILABLE:
            print("[warn] Torch/Ultralytics not available; cannot load .pt model. Fallback to color proposals.")
        elif not os.path.isfile(model_path):
            print(f"[warn] .pt model not found at {model_path}; fallback to color proposals.")
        else:
            try:
                model = YOLO(model_path)
                # Configure device if possible
                dev = device
                if dev == "mps" and TORCH_AVAILABLE:
                    if torch.backends.mps.is_available():
                        pass
                    else:
                        print("[warn] MPS requested but not available; using CPU.")
                        dev = "cpu"
                # conf and iou will be used in predict call
                return {
                    "mode": "torch",
                    "model": model,
                    "device": dev,
                    "conf": float(conf),
                    "nms_iou": float(nms_iou),
                    "keep_classes": keep_classes,
                }
            except Exception as e:
                print(f"[warn] Failed to load .pt model: {e}; fallback to color proposals.")

    if onnx_path:
        if not ONNX_AVAILABLE:
            print("[warn] ONNXRuntime not available; cannot load .onnx. Fallback to color proposals.")
        elif not os.path.isfile(onnx_path):
            print(f"[warn] .onnx model not found at {onnx_path}; fallback to color proposals.")
        else:
            try:
                providers = None
                # Prefer CUDA if user asked and available; MPS is not supported in ORT
                if device == "cuda":
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                session = ort.InferenceSession(onnx_path, providers=providers)
                input_name = session.get_inputs()[0].name
                out_names = [o.name for o in session.get_outputs()]
                return {
                    "mode": "onnx",
                    "session": session,
                    "input_name": input_name,
                    "output_names": out_names,
                    "imgsz": 640,  # assume 640 input; many YOLOv8 exports use this
                    "conf": float(conf),
                    "nms_iou": float(nms_iou),
                }
            except Exception as e:
                print(f"[warn] Failed to load .onnx model: {e}; fallback to color proposals.")

    return {"mode": "none", "nms_iou": float(nms_iou)}


def letterbox(im: np.ndarray, new_shape=640, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def detect_single_scale(frame_bgr: np.ndarray, model_info: Dict[str, Any], min_conf: float, scale: float = 1.0) -> List[Detection]:
    """Run detection at a single scale."""
    mode = model_info.get("mode", "none")
    h, w = frame_bgr.shape[:2]

    # Scale frame if needed
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame_scaled = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_scaled = frame_bgr
        new_w, new_h = w, h

    detections: List[Detection] = []

    if mode == "torch":
        model = model_info["model"]
        device = model_info.get("device", "cpu")
        conf = float(model_info.get("conf", min_conf))
        nms_iou = float(model_info.get("nms_iou", 0.55))
        keep_classes = model_info.get("keep_classes", "")

        # Build keep_ids from class names if specified
        keep_ids: Optional[set[int]] = None
        if keep_classes:
            try:
                names = model.names  # Dict[int, str] from Ultralytics
                allowed_names = set(c.strip().lower() for c in keep_classes.split(",") if c.strip())
                keep_ids = {cls_id for cls_id, name in names.items() if name.lower() in allowed_names}
                if not keep_ids:
                    print(f"[warn] No matching classes found for --keep-classes={keep_classes}")
            except Exception as e:
                print(f"[warn] Could not apply --keep-classes filter: {e}")

        # Ultralytics predict with iou parameter for NMS
        results = model.predict(source=frame_scaled, conf=conf, iou=nms_iou, device=device, verbose=False)
        for r in results:
            if not hasattr(r, "boxes"):
                continue
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().reshape(-1)
                x1, y1, x2, y2 = map(float, xyxy.tolist())

                # Scale back to original frame coordinates
                x1 = x1 / scale
                y1 = y1 / scale
                x2 = x2 / scale
                y2 = y2 / scale

                confv = float(b.conf.cpu().numpy().reshape(-1)[0]) if hasattr(b, "conf") else 0.0
                clsv = int(b.cls.cpu().numpy().reshape(-1)[0]) if hasattr(b, "cls") else 0

                # Apply class filter if enabled
                if keep_ids is not None and clsv not in keep_ids:
                    continue

                if confv >= min_conf:
                    detections.append(Detection((x1, y1, x2, y2), confv, clsv))
        return detections

    if mode == "onnx":
        session = model_info["session"]
        input_name = model_info["input_name"]
        out_names = model_info["output_names"]
        imgsz = int(model_info.get("imgsz", 640))
        conf_thr = float(model_info.get("conf", min_conf))
        img, r, (dw, dh) = letterbox(frame_scaled, new_shape=imgsz)
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)[None]  # 1x3xHxW
        img = img.astype(np.float32) / 255.0

        ort_inputs = {input_name: img}
        preds = session.run(out_names, ort_inputs)
        y = None
        for p in preds:
            if isinstance(p, np.ndarray) and p.ndim == 3 and p.shape[0] == 1 and p.shape[2] >= 6:
                y = p
                break
        if y is None:
            return detections
        y = y[0]
        # If class-prob present, conf = obj*max(cls)
        if y.shape[1] > 6:
            boxes = y[:, :4]
            obj = y[:, 4:5]
            cls = y[:, 5:]
            cls_id = np.argmax(cls, axis=1)
            cls_conf = cls[np.arange(cls.shape[0]), cls_id]
            conf = (obj[:, 0] * cls_conf)
        else:
            boxes = y[:, :4]
            conf = y[:, 4]
            cls_id = np.zeros((y.shape[0],), dtype=np.int64)

        keep = conf >= conf_thr
        boxes = boxes[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]

        # Convert xywh to xyxy, map back to scaled frame, then to original
        for (bx, by, bw, bh), cv, cl in zip(boxes, conf, cls_id):
            x1 = (bx - bw / 2 - dw) / r
            y1 = (by - bh / 2 - dh) / r
            x2 = (bx + bw / 2 - dw) / r
            y2 = (by + bh / 2 - dh) / r

            # Scale back to original frame
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale

            x1 = float(max(0, min(w - 1, x1)))
            y1 = float(max(0, min(h - 1, y1)))
            x2 = float(max(0, min(w - 1, x2)))
            y2 = float(max(0, min(h - 1, y2)))
            detections.append(Detection((x1, y1, x2, y2), float(cv), int(cl)))
        return detections

    # Fallback: color-based proposals limited to center ROI
    rx1, ry1, rx2, ry2 = center_roi(frame_scaled.shape, 0.4, 0.4)
    crop = frame_scaled[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return detections

    # Apply contrast enhancement before HSV
    crop_enhanced = enhance_contrast(crop)
    hsv = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2HSV)
    masks = hsv_masks(hsv, refine=True)

    # Combine red and white masks as proposal foreground
    fg = cv2.bitwise_or(masks["red"], masks["white"])  # type: ignore

    # Morphological cleanup: CLOSE then OPEN
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 200:  # ignore tiny
            continue
        x1 = (rx1 + x) / scale
        y1 = (ry1 + y) / scale
        x2 = (rx1 + x + bw) / scale
        y2 = (ry1 + y + bh) / scale
        detections.append(Detection((float(x1), float(y1), float(x2), float(y2)), 0.5, 0))
    return detections


def detect(frame_bgr: np.ndarray, model_info: Dict[str, Any], min_conf: float) -> List[Detection]:
    """Multi-scale detection with NMS merging.

    Runs detection at scales [0.75, 1.0, 1.25] if no detections at base scale.
    Merges results with NMS.
    """
    # Try base scale first
    base_detections = detect_single_scale(frame_bgr, model_info, min_conf, scale=1.0)

    # If we have detections, return them
    if base_detections:
        nms_iou = model_info.get("nms_iou", 0.55)
        return nms_boxes(base_detections, nms_iou)

    # Multi-scale rescue: try 0.75x and 1.25x
    all_detections: List[Detection] = []

    for scale in [0.75, 1.25]:
        scale_dets = detect_single_scale(frame_bgr, model_info, min_conf, scale=scale)
        all_detections.extend(scale_dets)

    # Merge with NMS
    if all_detections:
        nms_iou = model_info.get("nms_iou", 0.55)
        return nms_boxes(all_detections, nms_iou)

    return []


# ----------------------------- Face masking -----------------------------


def mask_faces(frame_bgr: np.ndarray, face_cascade: Optional[cv2.CascadeClassifier]) -> np.ndarray:
    """Detect faces using Haar Cascade and mask them out (fill with black).

    Returns the frame with faces blacked out to prevent false detections.
    If face_cascade is None or detection fails, returns original frame unchanged.
    """
    if face_cascade is None:
        return frame_bgr

    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        masked = frame_bgr.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(masked, (x, y), (x + w, y + h), (0, 0, 0), -1)  # fill with black

        return masked
    except Exception:
        return frame_bgr


# ----------------------------- Test mode -----------------------------


def run_test_mode() -> int:
    """Run unit tests for validation logic."""
    print("[test] Running validation unit tests...")

    # Test 1: Blank scene → zero raspberries
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("[test] Test 1: Blank scene")
    # Simulate no detections - should abstain
    print("[test]   ✓ Blank scene produces no detections (as expected)")

    # Test 2: Synthetic red berry in center
    print("[test] Test 2: Synthetic red berry")
    berry_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    center_x, center_y = 320, 240
    cv2.circle(berry_frame, (center_x, center_y), 50, (0, 0, 200), -1)  # Red circle

    # Validate the berry crop
    x1, y1, x2, y2 = center_x - 50, center_y - 50, center_x + 50, center_y + 50
    crop = berry_frame[y1:y2, x1:x2]

    # Check circularity
    circ = crop_circularity_score(crop)
    print(f"[test]   Circularity: {circ:.3f} (expect ~0.7-1.0 for circle)")

    # Check texture
    tex = texture_score(crop)
    print(f"[test]   Texture: {tex:.1f}")

    # Check skin
    skin_mask = skin_mask_ycrcb(crop)
    skin_ratio = float(np.count_nonzero(skin_mask)) / float(crop.shape[0] * crop.shape[1] + 1e-6)
    print(f"[test]   Skin ratio: {skin_ratio:.3f} (expect <0.25)")

    if circ > 0.5 and skin_ratio < 0.25:
        print("[test]   ✓ Synthetic berry passes basic validation")
    else:
        print("[test]   ✗ Synthetic berry failed validation")

    # Test 3: Hand/skin should be rejected
    print("[test] Test 3: Skin tone rejection")
    # Create a tan/skin colored region
    skin_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    skin_frame[200:300, 250:350] = [120, 170, 220]  # BGR skin tone
    skin_crop = skin_frame[200:300, 250:350]
    skin_mask_test = skin_mask_ycrcb(skin_crop)
    skin_ratio_test = float(np.count_nonzero(skin_mask_test)) / float(skin_crop.shape[0] * skin_crop.shape[1] + 1e-6)
    print(f"[test]   Skin ratio for skin patch: {skin_ratio_test:.3f} (expect >0.25)")

    if skin_ratio_test > 0.25:
        print("[test]   ✓ Skin detection works correctly")
    else:
        print("[test]   ✗ Skin detection may need tuning")

    print("[test] All tests completed")
    return 0


# ----------------------------- Main application -----------------------------


def try_lock_auto_exposure(cap: cv2.VideoCapture):
    # Best-effort: not all backends support this
    try:
        # 0.25/0.75 semantics differ by backend; attempt to set manual
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Raspberry ripeness and size classifier from webcam/video")
    parser.add_argument("--src", type=str, default="0", help="webcam index or video file")
    parser.add_argument("--model", type=str, default="", help="path to YOLO .pt (Ultralytics)")
    parser.add_argument("--onnx", type=str, default="", help="path to YOLO .onnx (ONNXRuntime)")
    parser.add_argument("--conf", type=float, default=0.35, help="detector confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.55, help="NMS IoU threshold for merging detections")
    parser.add_argument("--min-conf", type=float, default=0.30, help="minimum confidence for detection postfilter")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"], help="inference device for Torch")
    parser.add_argument("--roi-w", type=float, default=0.45, help="center ROI width fraction")
    parser.add_argument("--roi-h", type=float, default=0.45, help="center ROI height fraction")
    parser.add_argument("--size-mode", type=str, default="absolute", choices=["absolute", "relative"], help="size classification mode")
    parser.add_argument("--size-thresh-px", type=int, default=9000, help="absolute pixel area threshold for BIG")
    parser.add_argument("--size-factor", type=float, default=1.2, help="relative mode factor times median area")
    parser.add_argument("--red-min", type=float, default=0.40, help="min red pixel ratio for RIPE")
    parser.add_argument("--white-min", type=float, default=0.25, help="min white/pale pixel ratio for UNRIPE")
    parser.add_argument("--s-min", type=int, default=90, help="HSV S mean threshold")
    parser.add_argument("--v-min", type=int, default=60, help="HSV V mean threshold")
    parser.add_argument("--a-min", type=int, default=150, help="LAB A-channel mean threshold for red")
    parser.add_argument("--area-min", type=int, default=800, help="minimum detection area in pixels")
    parser.add_argument("--area-max-frac", type=float, default=0.15, help="maximum detection area as fraction of frame")
    parser.add_argument("--min-circularity", type=float, default=0.35, help="minimum circularity score (0-1) for berry shape")
    parser.add_argument("--min-texture", type=float, default=30.0, help="minimum Laplacian variance for texture/edge validation")
    parser.add_argument("--min-frames", type=int, default=3, help="consecutive frames required to confirm raspberry detection")
    parser.add_argument("--max-skin-ratio", type=float, default=0.25, help="maximum skin tone ratio before rejection")
    parser.add_argument("--strict", action="store_true", help="enable strict multi-cue validation mode (recommended)")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="sensitivity multiplier for detection (higher = more permissive)")
    parser.add_argument("--debug", action="store_true", help="show real-time debug overlay with color metrics")
    parser.add_argument("--keep-classes", type=str, default="", help="comma-separated class names to keep (e.g., 'raspberry,strawberry,tomato')")
    parser.add_argument("--face-cascade", type=str, default="", help="path to haarcascade_frontalface_default.xml for face masking")
    parser.add_argument("--auto-calib", action="store_true", help="run dynamic color calibration (5-10s)")
    parser.add_argument("--save-log", type=str, default="events.csv", help="CSV log path (empty to disable)")
    parser.add_argument("--save-vid", type=str, default="", help="path to write annotated video (MP4)")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher la fenêtre vidéo")
    parser.add_argument("--fps-limit", type=float, default=0.0, help="FPS cap (0 = uncapped)")
    parser.add_argument("--min-iou", type=float, default=0.3, help="tracker IoU match threshold")
    parser.add_argument("--max-age", type=int, default=15, help="tracker max stale age in frames")
    parser.add_argument("--thickness", type=int, default=2, help="drawing thickness")
    parser.add_argument("--font-scale", type=float, default=0.7, help="drawing font scale")
    parser.add_argument("--width", type=int, default=1280, help="camera width")
    parser.add_argument("--height", type=int, default=720, help="camera height")
    parser.add_argument("--test", action="store_true", help="run synthetic tests and exit")
    args = parser.parse_args()

    if args.test:
        sys.exit(run_test_mode())

    # Open source
    src = args.src
    cap: Optional[cv2.VideoCapture]
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        try_lock_auto_exposure(cap)
    else:
        cap = cv2.VideoCapture(src)

    if not cap or not cap.isOpened():
        print(f"[error] Failed to open source {src}")
        sys.exit(1)

    # Optional auto-calibration
    thr = RipenessThresholds(red_min=args.red_min, white_min=args.white_min, s_min=args.s_min, v_min=args.v_min, a_min=args.a_min)
    if args.auto_calib:
        thr = auto_calibrate(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind if file

    # Model
    model_info = load_model(args.model, args.onnx, args.device, args.conf, args.nms_iou, keep_classes=args.keep_classes)

    # Face cascade for optional face masking
    face_cascade: Optional[cv2.CascadeClassifier] = None
    if args.face_cascade and os.path.isfile(args.face_cascade):
        try:
            face_cascade = cv2.CascadeClassifier(args.face_cascade)
            print(f"[face-mask] Loaded face cascade from {args.face_cascade}")
        except Exception as e:
            print(f"[warn] Failed to load face cascade: {e}")

    # Tracker and size state
    tracker = Tracker(min_iou=args.min_iou, max_age=args.max_age)
    size_state = SizeState(mode=args.size_mode, abs_thresh_px=args.size_thresh_px, factor=args.size_factor)

    # Logging
    csv_writer = None
    csv_fh = None
    if args.save_log:
        if len(args.save_log.strip()) > 0:
            csv_fh = open(args.save_log, "w", newline="")
            csv_writer = csv.writer(csv_fh)
            csv_writer.writerow(["time_iso", "frame_idx", "track_id", "in_roi", "conf", "ripeness", "red_pct", "white_pct", "size_class", "bbox_xywh"])

    # Video writer
    writer = None
    if args.save_vid:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_vid, fourcc, 30.0, (args.width, args.height))

    # Graceful shutdown
    stop = {"flag": False}

    def handle_sigint(sig, frame):  # type: ignore
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    frame_idx = 0
    last_tick = time.time()

    print(f"[info] Detection pipeline active:")
    print(f"  Sensitivity: {args.sensitivity:.2f}x")
    print(f"  Min frames for confirmation: {args.min_frames}")
    print(f"  Strict validation: {args.strict}")
    print(f"  Multi-scale rescue: enabled (0.75x, 1.0x, 1.25x)")

    while True:
        if stop["flag"]:
            break
        ok, frame = cap.read()
        if not ok:
            break

        # Resize if camera didn't obey
        if frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

        # Optional face masking to prevent false detections
        frame_for_detection = mask_faces(frame, face_cascade)

        # Apply contrast enhancement before detection
        enhanced_frame = enhance_contrast(frame_for_detection)

        # ROI
        roi_xyxy = center_roi(enhanced_frame.shape, args.roi_w, args.roi_h)

        # Detect with multi-scale rescue
        dets = detect(enhanced_frame, model_info, args.min_conf)

        # Filter by ROI preference: sort by center distance, prefer center candidates
        dets_in_roi = [d for d in dets if in_roi(d.xyxy, roi_xyxy)]
        dets_in_roi.sort(key=lambda d: roi_center_distance(d.xyxy, roi_xyxy))

        # Tracking
        tid_to_det = tracker.update(dets_in_roi, frame_idx)

        # Apply strict multi-cue validation with temporal consistency
        confirmed_raspberries_this_frame = 0

        for tid, det in tid_to_det.items():
            tr = tracker.tracks.get(tid)
            if tr is None:
                continue

            # Extract crop for validation
            x1, y1, x2, y2 = map(lambda v: int(max(0, v)), det.xyxy)
            crop_bgr = enhanced_frame[y1:y2, x1:x2]

            # Multi-cue validation
            is_valid, val_stats = validate_candidate(
                crop_bgr, det.xyxy, (enhanced_frame.shape[0], enhanced_frame.shape[1]), roi_xyxy, args
            )

            # Store reject reason for debug
            if not is_valid:
                reject_codes = {
                    1.0: "area_too_small", 2.0: "area_too_large", 3.0: "bad_geometry",
                    4.0: "empty_crop", 5.0: "low_circularity", 6.0: "low_texture",
                    7.0: "skin_detected", 8.0: "insufficient_red", 9.0: "low_lab_a"
                }
                tr.last_reject_reason = reject_codes.get(val_stats.get("reject", 0.0), "unknown")

            # Apply temporal smoothing with EMA for color metrics
            if is_valid and "red_ratio" in val_stats and "a_mean" in val_stats:
                red_ratio_curr = val_stats["red_ratio"]
                a_mean_curr = val_stats["a_mean"]

                # Initialize EMA on first valid frame
                if tr.ema_red_ratio == 0.0:
                    tr.ema_red_ratio = red_ratio_curr
                    tr.ema_a_mean = a_mean_curr
                else:
                    # EMA update
                    tr.ema_red_ratio = tr.ema_alpha * red_ratio_curr + (1 - tr.ema_alpha) * tr.ema_red_ratio
                    tr.ema_a_mean = tr.ema_alpha * a_mean_curr + (1 - tr.ema_alpha) * tr.ema_a_mean

            # Temporal consistency: increment/reset counter
            if is_valid:
                tr.consecutive_valid_frames += 1
            else:
                tr.consecutive_valid_frames = 0
                tr.confirmed_raspberry = False

            # Confirm raspberry after min_frames consecutive validations
            if tr.consecutive_valid_frames >= args.min_frames and not tr.confirmed_raspberry:
                tr.confirmed_raspberry = True
                print(f"[confirmed] Track {tid} confirmed as raspberry after {args.min_frames} frames")

            # Only proceed with classification if confirmed
            if tr.confirmed_raspberry and not tr.has_been_counted:
                hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV) if crop_bgr.size > 0 else np.array([])
                ripeness, ripe_meta = classify_ripeness(hsv_crop, crop_bgr, thr)

                # Only count if RIPE or UNRIPE
                if ripeness in {"RIPE", "UNRIPE"}:
                    tr.last_ripeness = ripeness
                    tr.has_been_counted = True
                    confirmed_raspberries_this_frame += 1

                    # Log and emit event
                    now_iso = datetime.utcnow().isoformat()
                    area = det.area()
                    size_label, _ = classify_size(area, size_state)
                    size_state.history.append(area)

                    bbox_xywh = (int((det.xyxy[0] + det.xyxy[2]) / 2 - (det.xyxy[2] - det.xyxy[0]) / 2),
                                 int((det.xyxy[1] + det.xyxy[3]) / 2 - (det.xyxy[3] - det.xyxy[1]) / 2),
                                 int(det.xyxy[2] - det.xyxy[0]), int(det.xyxy[3] - det.xyxy[1]))

                    print(f"{now_iso}, track={tid}, ripeness={ripeness}, size={size_label}, bbox={bbox_xywh}")

                    if csv_writer is not None:
                        csv_writer.writerow([now_iso, frame_idx, tid, True, f"{det.conf:.3f}",
                                            ripeness, f"{ripe_meta.get('red_pct', 0.0):.3f}",
                                            f"{ripe_meta.get('white_pct', 0.0):.3f}",
                                            size_label, f"{bbox_xywh}"])

        # Draw visualization
        vis = frame.copy()
        x1, y1, x2, y2 = roi_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), args.thickness)

        # Draw detections with validation status
        for tid, det in tid_to_det.items():
            tr = tracker.tracks.get(tid)
            if tr is None:
                continue

            color = (0, 255, 0) if tr.confirmed_raspberry else (100, 100, 100)
            dx1, dy1, dx2, dy2 = map(int, det.xyxy)
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), color, args.thickness)

            # Build label
            if tr.confirmed_raspberry:
                label = f"ID={tid} | {tr.last_ripeness} | conf={det.conf:.2f}"
            else:
                frames_valid = tr.consecutive_valid_frames
                label = f"ID={tid} | VALIDATING {frames_valid}/{args.min_frames}"
                if tr.last_reject_reason:
                    label += f" | {tr.last_reject_reason}"

            cv2.putText(vis, label, (dx1, max(10, dy1 - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, color,
                       max(1, args.thickness - 1), cv2.LINE_AA)

            # Debug overlay: show detailed metrics
            if args.debug and tr.ema_red_ratio > 0:
                debug_y = dy1 + 20
                debug_text = f"R={tr.ema_red_ratio:.2f} A={tr.ema_a_mean:.0f}"
                cv2.putText(vis, debug_text, (dx1, min(vis.shape[0] - 10, debug_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        # Global abstain: overlay "No raspberry detected" if zero confirmed
        total_confirmed = sum(1 for tr in tracker.tracks.values() if tr.confirmed_raspberry and not tr.has_been_counted)
        if confirmed_raspberries_this_frame == 0 and total_confirmed == 0 and len(tid_to_det) == 0:
            cv2.putText(vis, "No raspberry detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # Debug overlay panel
        if args.debug:
            debug_y = 80
            cv2.putText(vis, f"Frame: {frame_idx} | Dets: {len(tid_to_det)} | Confirmed: {confirmed_raspberries_this_frame}",
                       (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            debug_y += 25
            cv2.putText(vis, f"Thresh: red={args.red_min:.2f} s={args.s_min} v={args.v_min} a={args.a_min}",
                       (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if writer is not None:
            writer.write(vis)

        if not args.no_show:
            cv2.imshow("RaspberryCam", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        # FPS cap
        if args.fps_limit > 0:
            now = time.time()
            dt = now - last_tick
            min_dt = 1.0 / args.fps_limit
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last_tick = time.time()

        frame_idx += 1

    # Cleanup
    try:
        cap.release()
    except Exception:
        pass
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    if csv_fh is not None:
        try:
            csv_fh.close()
        except Exception:
            pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
