#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour diviser le dataset de framboises en train/valid/test splits.
Usage: python split_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset(
    source_dir: str = "Raspberry/train",
    output_base: str = "Raspberry",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Divise le dataset en train/valid/test splits.

    Args:
        source_dir: Dossier source contenant images/ et labels/
        output_base: Dossier de base pour la sortie
        train_ratio: Proportion pour l'entraînement
        valid_ratio: Proportion pour la validation
        test_ratio: Proportion pour le test
        seed: Seed pour la reproductibilité
    """
    random.seed(seed)

    source_images = Path(source_dir) / "images"
    source_labels = Path(source_dir) / "labels"

    # Vérifier que les dossiers source existent
    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"Dossiers source non trouvés: {source_images} ou {source_labels}")

    # Lister toutes les images
    image_files = sorted([f for f in os.listdir(source_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[info] Trouvé {len(image_files)} images dans {source_images}")

    # Mélanger aléatoirement
    random.shuffle(image_files)

    # Calculer les splits
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid

    splits = {
        "train": image_files[:n_train],
        "valid": image_files[n_train:n_train + n_valid],
        "test": image_files[n_train + n_valid:]
    }

    print(f"[split] Train: {len(splits['train'])}, Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")

    # Créer les dossiers de destination
    output_base_path = Path(output_base)
    for split_name in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            dest_dir = output_base_path / split_name / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)

    # Copier les fichiers dans les splits appropriés
    for split_name, file_list in splits.items():
        print(f"\n[{split_name}] Copie de {len(file_list)} fichiers...")

        for img_file in file_list:
            # Image
            src_img = source_images / img_file
            dst_img = output_base_path / split_name / "images" / img_file
            shutil.copy2(src_img, dst_img)

            # Label (même nom mais .txt)
            label_file = Path(img_file).stem + ".txt"
            src_label = source_labels / label_file
            dst_label = output_base_path / split_name / "labels" / label_file

            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                print(f"[warn] Label manquant pour {img_file}")

    print(f"\n[done] Dataset divisé avec succès!")
    print(f"  Train: {output_base_path}/train")
    print(f"  Valid: {output_base_path}/valid")
    print(f"  Test: {output_base_path}/test")


if __name__ == "__main__":
    # Créer une sauvegarde du dossier train original
    if not os.path.exists("Raspberry/train_backup"):
        print("[backup] Sauvegarde du dossier train original...")
        shutil.copytree("Raspberry/train", "Raspberry/train_backup")
        print("[backup] Sauvegarde créée: Raspberry/train_backup")

    # Diviser le dataset
    split_dataset(
        source_dir="Raspberry/train_backup",
        output_base="Raspberry",
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
