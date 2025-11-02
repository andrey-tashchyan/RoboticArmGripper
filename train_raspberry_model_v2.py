#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement YOLOv8 pour la détection de framboises avec barre de progression.

Usage:
    # Entraînement rapide (pour tester):
    python train_raspberry_model_v2.py --epochs 10 --imgsz 416

    # Entraînement complet:
    python train_raspberry_model_v2.py --epochs 100 --imgsz 640 --batch 16

    # Entraînement avec GPU MPS (Apple Silicon):
    python train_raspberry_model_v2.py --epochs 100 --device mps
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    from ultralytics.utils import callbacks
except ImportError:
    print("[error] Ultralytics n'est pas installé. Installez-le avec: pip install ultralytics")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("[warn] tqdm n'est pas installé. Installez-le pour une meilleure barre de progression: pip install tqdm")
    tqdm = None


class TrainingProgressTracker:
    """Tracker de progression d'entraînement avec affichage en temps réel."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.best_map = 0.0
        self.last_loss = 0.0

        # Barre de progression principale
        if tqdm:
            self.pbar = tqdm(total=total_epochs, desc="🍓 Entraînement",
                           unit="epoch", ncols=100,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            self.pbar = None

    def on_train_epoch_start(self, trainer):
        """Appelé au début de chaque époque."""
        self.current_epoch = trainer.epoch
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer):
        """Appelé à la fin de chaque époque."""
        metrics = trainer.metrics
        epoch_time = time.time() - self.epoch_start_time

        # Récupérer les métriques
        box_loss = metrics.get('train/box_loss', 0.0)
        cls_loss = metrics.get('train/cls_loss', 0.0)
        dfl_loss = metrics.get('train/dfl_loss', 0.0)
        total_loss = box_loss + cls_loss + dfl_loss

        self.last_loss = total_loss

        # Afficher les métriques
        if self.pbar:
            self.pbar.set_postfix({
                'loss': f'{total_loss:.4f}',
                'time': f'{epoch_time:.1f}s'
            })
            self.pbar.update(1)
        else:
            elapsed = time.time() - self.start_time
            print(f"\r[Epoch {self.current_epoch + 1}/{self.total_epochs}] "
                  f"Loss: {total_loss:.4f} | "
                  f"Temps: {epoch_time:.1f}s | "
                  f"Total: {elapsed/60:.1f}min", end='', flush=True)

    def on_fit_epoch_end(self, trainer):
        """Appelé après validation de chaque époque."""
        metrics = trainer.metrics

        # Récupérer mAP
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)

        if map50 > self.best_map:
            self.best_map = map50

        if self.pbar:
            self.pbar.set_postfix({
                'loss': f'{self.last_loss:.4f}',
                'mAP50': f'{map50:.3f}',
                'best': f'{self.best_map:.3f}'
            })
        else:
            print(f" | mAP50: {map50:.3f} (best: {self.best_map:.3f})", flush=True)

    def on_train_end(self, trainer):
        """Appelé à la fin de l'entraînement."""
        total_time = time.time() - self.start_time

        if self.pbar:
            self.pbar.close()

        print("\n")
        print("=" * 80)
        print("✅ ENTRAÎNEMENT TERMINÉ!")
        print("=" * 80)
        print(f"⏱️  Durée totale: {total_time/60:.1f} minutes")
        print(f"🎯 Meilleur mAP50: {self.best_map:.4f}")
        print("=" * 80)


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
):
    """Entraîne un modèle YOLOv8 pour la détection de framboises avec barre de progression."""

    print("=" * 80)
    print("🍓 ENTRAÎNEMENT YOLOV8 - DÉTECTION DE FRAMBOISES")
    print("=" * 80)
    print(f"📦 Modèle de base: {model_name}")
    print(f"📊 Dataset: {data_yaml}")
    print(f"🔄 Époques: {epochs}")
    print(f"📐 Taille d'image: {imgsz}")
    print(f"📦 Batch size: {batch}")
    print(f"💻 Device: {device}")
    print(f"⚙️  Optimiseur: {optimizer}")
    print(f"📈 Learning rate: {lr0}")
    print(f"🎨 Augmentation: {'✅ Activée' if augment else '❌ Désactivée'}")
    print("=" * 80)
    print()

    # Vérifier que le fichier data.yaml existe
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Fichier data.yaml non trouvé: {data_yaml}")

    # Charger le modèle
    print(f"🔧 Chargement du modèle {model_name}...")
    model = YOLO(model_name)

    # Créer le tracker de progression
    progress_tracker = TrainingProgressTracker(epochs)

    # Ajouter les callbacks personnalisés
    model.add_callback("on_train_epoch_start", progress_tracker.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", progress_tracker.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", progress_tracker.on_fit_epoch_end)
    model.add_callback("on_train_end", progress_tracker.on_train_end)

    # Configuration de l'entraînement
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = f"{save_dir}_{timestamp}"

    print(f"🚀 Démarrage de l'entraînement...")
    print(f"💾 Résultats: {project_dir}\n")

    # Entraînement
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        save=True,
        save_period=10,
        project=save_dir,
        name=f"train_{timestamp}",
        exist_ok=True,
        pretrained=pretrained,
        optimizer=optimizer,
        verbose=False,  # Désactiver verbose pour éviter les conflits avec notre barre
        seed=42,
        deterministic=True,
        single_cls=True,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        profile=False,
        # Learning rate
        lr0=lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=weight_decay,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Augmentations
        hsv_h=0.015 if augment else 0.0,
        hsv_s=0.7 if augment else 0.0,
        hsv_v=0.4 if augment else 0.0,
        degrees=10.0 if augment else 0.0,
        translate=0.2 if augment else 0.0,
        scale=0.5 if augment else 0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5 if augment else 0.0,
        mosaic=1.0 if augment else 0.0,
        mixup=0.1 if augment else 0.0,
        copy_paste=0.0,
    )

    # Afficher les métriques finales
    print()
    print("=" * 80)
    print("📊 MÉTRIQUES FINALES")
    print("=" * 80)

    final_metrics = results.results_dict
    print(f"🎯 mAP50: {final_metrics.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"🎯 mAP50-95: {final_metrics.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f"✓  Precision: {final_metrics.get('metrics/precision(B)', 0.0):.4f}")
    print(f"✓  Recall: {final_metrics.get('metrics/recall(B)', 0.0):.4f}")

    # Chemin du meilleur modèle
    best_model_path = Path(save_dir) / f"train_{timestamp}" / "weights" / "best.pt"
    print()
    print("=" * 80)
    print("💾 FICHIERS GÉNÉRÉS")
    print("=" * 80)
    print(f"⭐ Meilleur modèle: {best_model_path}")
    print(f"📝 Dernier modèle: {Path(save_dir) / f'train_{timestamp}' / 'weights' / 'last.pt'}")
    print(f"📊 Résultats: {Path(save_dir) / f'train_{timestamp}' / 'results.csv'}")
    print(f"📈 Graphiques: {Path(save_dir) / f'train_{timestamp}' / 'results.png'}")

    # Validation sur le test set
    print()
    print("=" * 80)
    print("🧪 VALIDATION SUR LE TEST SET")
    print("=" * 80)
    test_results = model.val(data=data_yaml, split='test')
    print(f"🎯 Test mAP50: {test_results.box.map50:.4f}")
    print(f"🎯 Test mAP50-95: {test_results.box.map:.4f}")

    print()
    print("=" * 80)
    print("🚀 UTILISATION DU MODÈLE")
    print("=" * 80)
    print(f"Pour utiliser ce modèle avec raspberry_cam.py:")
    print()
    print(f"  source .venv311/bin/activate")
    print(f"  python raspberry_cam.py \\")
    print(f"      --src 0 \\")
    print(f"      --model {best_model_path} \\")
    print(f"      --device {device} \\")
    print(f"      --strict \\")
    print(f"      --sensitivity 1.5 \\")
    print(f"      --auto-calib \\")
    print(f"      --debug")
    print()
    print("=" * 80)

    return str(best_model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement YOLOv8 pour la détection de framboises avec barre de progression",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Modèle de base (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
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
