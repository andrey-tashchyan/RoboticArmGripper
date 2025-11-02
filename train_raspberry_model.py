#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement YOLOv8 pour la détection de framboises.

Usage:
    # Entraînement rapide (pour tester):
    python train_raspberry_model.py --epochs 10 --imgsz 416

    # Entraînement complet:
    python train_raspberry_model.py --epochs 100 --imgsz 640 --batch 16

    # Entraînement avec GPU MPS (Apple Silicon):
    python train_raspberry_model.py --epochs 100 --device mps

    # Fine-tuning d'un modèle pré-entraîné:
    python train_raspberry_model.py --model yolov8s.pt --epochs 50
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("[error] Ultralytics n'est pas installé. Installez-le avec: pip install ultralytics")
    sys.exit(1)


def train_raspberry_model(
    model_name: str = "yolov8n.pt",
    data_yaml: str = "Raspberry/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "mps",
    patience: int = 20,
    save_dir: str = "runs/raspberry_detect",
    pretrained: bool = True,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    weight_decay: float = 0.0005,
    augment: bool = True,
    mosaic: float = 1.0,
    mixup: float = 0.1,
    degrees: float = 10.0,
    translate: float = 0.2,
    scale: float = 0.5,
    fliplr: float = 0.5,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
):
    """Entraîne un modèle YOLOv8 pour la détection de framboises.

    Args:
        model_name: Nom du modèle de base (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        data_yaml: Chemin vers le fichier data.yaml
        epochs: Nombre d'époques d'entraînement
        imgsz: Taille des images (640, 512, 416, etc.)
        batch: Taille du batch
        device: Device (mps, cuda, cpu)
        patience: Early stopping patience
        save_dir: Dossier de sauvegarde des résultats
        pretrained: Utiliser les poids pré-entraînés sur COCO
        optimizer: Optimiseur (AdamW, SGD, Adam)
        lr0: Learning rate initial
        weight_decay: Poids de régularisation L2
        augment: Activer les augmentations de données
        mosaic: Probabilité d'augmentation mosaic
        mixup: Probabilité d'augmentation mixup
        degrees: Rotation maximale en degrés
        translate: Translation maximale (fraction de l'image)
        scale: Scaling maximal (±)
        fliplr: Probabilité de flip horizontal
        hsv_h: Augmentation de teinte HSV
        hsv_s: Augmentation de saturation HSV
        hsv_v: Augmentation de valeur HSV
    """
    print("=" * 80)
    print("ENTRAÎNEMENT YOLOV8 - DÉTECTION DE FRAMBOISES")
    print("=" * 80)
    print(f"[config] Modèle de base: {model_name}")
    print(f"[config] Dataset: {data_yaml}")
    print(f"[config] Époques: {epochs}")
    print(f"[config] Taille d'image: {imgsz}")
    print(f"[config] Batch size: {batch}")
    print(f"[config] Device: {device}")
    print(f"[config] Optimiseur: {optimizer}")
    print(f"[config] Learning rate: {lr0}")
    print(f"[config] Augmentation: {augment}")
    print("=" * 80)

    # Vérifier que le fichier data.yaml existe
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Fichier data.yaml non trouvé: {data_yaml}")

    # Charger le modèle
    print(f"\n[model] Chargement du modèle {model_name}...")
    model = YOLO(model_name)

    # Configuration de l'entraînement
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = f"{save_dir}_{timestamp}"

    print(f"\n[train] Démarrage de l'entraînement...")
    print(f"[train] Les résultats seront sauvegardés dans: {project_dir}\n")

    # Entraînement
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        save=True,
        save_period=10,  # Sauvegarder tous les 10 époques
        project=save_dir,
        name=f"train_{timestamp}",
        exist_ok=True,
        pretrained=pretrained,
        optimizer=optimizer,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # Une seule classe (Raspberry)
        rect=False,  # Rectangular training (peut accélérer)
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Désactiver mosaic dans les 10 dernières époques
        amp=True,  # Automatic Mixed Precision (plus rapide sur GPU)
        fraction=1.0,  # Utiliser 100% du dataset
        profile=False,  # Ne pas profiler
        # Learning rate et régularisation
        lr0=lr0,
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=weight_decay,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Loss weights
        box=7.5,  # Box loss gain
        cls=0.5,  # Classification loss gain (moins important, 1 classe)
        dfl=1.5,  # Distribution Focal Loss gain
        # Data augmentation (si augment=True)
        hsv_h=hsv_h if augment else 0.0,
        hsv_s=hsv_s if augment else 0.0,
        hsv_v=hsv_v if augment else 0.0,
        degrees=degrees if augment else 0.0,
        translate=translate if augment else 0.0,
        scale=scale if augment else 0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,  # Pas de flip vertical (framboises)
        fliplr=fliplr if augment else 0.0,
        mosaic=mosaic if augment else 0.0,
        mixup=mixup if augment else 0.0,
        copy_paste=0.0,
    )

    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT TERMINÉ!")
    print("=" * 80)

    # Afficher les métriques finales
    final_epoch = results.results_dict
    print(f"\n[metrics] Métriques finales (époque {epochs}):")
    print(f"  mAP50: {final_epoch.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {final_epoch.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision: {final_epoch.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall: {final_epoch.get('metrics/recall(B)', 'N/A'):.4f}")

    # Chemin du meilleur modèle
    best_model_path = Path(save_dir) / f"train_{timestamp}" / "weights" / "best.pt"
    print(f"\n[save] Meilleur modèle sauvegardé: {best_model_path}")
    print(f"[save] Dernier modèle: {Path(save_dir) / f'train_{timestamp}' / 'weights' / 'last.pt'}")

    # Validation sur le test set
    print("\n[val] Validation sur le test set...")
    test_results = model.val(data=data_yaml, split='test')
    print(f"[test] mAP50: {test_results.box.map50:.4f}")
    print(f"[test] mAP50-95: {test_results.box.map:.4f}")

    print("\n" + "=" * 80)
    print(f"Pour utiliser ce modèle avec raspberry_cam.py:")
    print(f"  python raspberry_cam.py --src 0 --model {best_model_path} --device {device}")
    print("=" * 80 + "\n")

    return str(best_model_path)


def main():
    parser = argparse.ArgumentParser(description="Entraînement YOLOv8 pour la détection de framboises")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Modèle de base (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)")
    parser.add_argument("--data", type=str, default="Raspberry/data.yaml",
                        help="Chemin vers data.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Nombre d'époques")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Taille des images (640, 512, 416)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Taille du batch")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps, cuda, cpu)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--save-dir", type=str, default="runs/raspberry_detect",
                        help="Dossier de sauvegarde")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Ne pas utiliser les poids pré-entraînés")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW", "Adam", "SGD"],
                        help="Optimiseur")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate initial")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay (L2)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Désactiver les augmentations de données")

    args = parser.parse_args()

    # Entraîner le modèle
    best_model = train_raspberry_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        save_dir=args.save_dir,
        pretrained=not args.no_pretrained,
        optimizer=args.optimizer,
        lr0=args.lr,
        weight_decay=args.weight_decay,
        augment=not args.no_augment,
    )

    return best_model


if __name__ == "__main__":
    main()
