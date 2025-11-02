#!/usr/bin/env python3
"""
🌈 Préparation du dataset YOLOv8 avec BARRES DE CHARGEMENT MULTICOLORES 🌈
Version ULTRA VISUELLE avec maximum de feedback !
"""

import os
import shutil
import time
from pathlib import Path
from tqdm import tqdm

# Codes couleur ANSI pour terminal
class Colors:
    HEADER = '\033[95m'      # Magenta
    OKBLUE = '\033[94m'      # Bleu
    OKCYAN = '\033[96m'      # Cyan
    OKGREEN = '\033[92m'     # Vert
    WARNING = '\033[93m'     # Jaune
    FAIL = '\033[91m'        # Rouge
    ENDC = '\033[0m'         # Reset
    BOLD = '\033[1m'         # Gras
    UNDERLINE = '\033[4m'    # Souligné

# Source directories
SCRIPT_DIR = Path(__file__).parent.resolve()
RASPBERRY_DIR = SCRIPT_DIR / "Raspberry"
BACKGROUND_DIR = SCRIPT_DIR / "background"
TARGET_DIR = SCRIPT_DIR / "data" / "raspberries"

def print_colored(text, color):
    """Affiche du texte coloré."""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text, char="="):
    """Affiche un header stylé."""
    line = char * 80
    print(f"\n{Colors.HEADER}{Colors.BOLD}{line}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{line}{Colors.ENDC}\n")

def animated_spinner(duration=0.5):
    """Animation de spinner."""
    frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        print(f'\r{Colors.OKCYAN}  {frames[i % len(frames)]} Chargement...{Colors.ENDC}', end='', flush=True)
        time.sleep(0.1)
        i += 1
    print('\r' + ' ' * 50 + '\r', end='', flush=True)

def create_progress_bar(current, total, prefix='', color=Colors.OKGREEN, width=40):
    """Crée une barre de progression colorée."""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    return f"{color}{prefix} |{bar}| {percent:.1f}% ({current}/{total}){Colors.ENDC}"

def scan_directory_with_animation(directory, description):
    """Scanne un dossier avec animation."""
    print_colored(f"  🔍 Scanning {description}...", Colors.OKCYAN)
    animated_spinner(0.3)

    if not directory.exists():
        print_colored(f"  ⚠️  Directory not found: {directory}", Colors.WARNING)
        return []

    # Scan avec barre de progression
    print_colored(f"  📂 Reading files from {directory.name}/", Colors.OKBLUE)
    image_files = []

    # Simuler le scan avec barre
    for pattern in ['*.jpg', '*.png']:
        files = list(directory.glob(pattern))
        image_files.extend(files)

    print_colored(f"  ✅ Found {len(image_files)} images!", Colors.OKGREEN)
    return image_files

def main():
    # HEADER ULTRA STYLÉ
    print_header("🍓 PRÉPARATION DATASET YOLOV8 - VERSION MULTICOLORE 🍓", "━")

    print_colored("🎨 Mode: MAXIMUM VISUAL FEEDBACK", Colors.HEADER)
    print_colored(f"📍 Working directory: {SCRIPT_DIR}", Colors.OKBLUE)
    print()

    # VÉRIFICATION DES DOSSIERS
    print_header("🔍 ÉTAPE 1/5 - VÉRIFICATION DES SOURCES", "─")

    sources = {
        'Raspberry (positives)': RASPBERRY_DIR,
        'Background (negatives)': BACKGROUND_DIR
    }

    for name, path in sources.items():
        if path.exists():
            print_colored(f"  ✅ {name}: {path}", Colors.OKGREEN)
        else:
            print_colored(f"  ❌ {name}: NOT FOUND - {path}", Colors.FAIL)

    # CRÉATION DES DOSSIERS CIBLES
    print_header("📁 ÉTAPE 2/5 - CRÉATION DES DOSSIERS", "─")

    splits = ['train', 'val', 'test']
    total_dirs = len(splits) * 2
    created = 0

    print_colored("  🏗️  Creating directory structure...", Colors.OKCYAN)
    for split in splits:
        (TARGET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        created += 1
        print(create_progress_bar(created, total_dirs, "  Images", Colors.OKBLUE))

        (TARGET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        created += 1
        print(create_progress_bar(created, total_dirs, "  Labels", Colors.HEADER))

    print_colored("  ✅ Tous les dossiers créés!", Colors.OKGREEN)

    # TRAITEMENT DES DONNÉES
    print_header("🍓 ÉTAPE 3/5 - COPIE DES IMAGES POSITIVES", "─")

    total_positives = 0
    splits_data = [('train', 'train'), ('valid', 'val'), ('test', 'test')]

    for src_split, dst_split in splits_data:
        src_images = RASPBERRY_DIR / src_split / "images"
        src_labels = RASPBERRY_DIR / src_split / "labels"
        dst_images = TARGET_DIR / "images" / dst_split
        dst_labels = TARGET_DIR / "labels" / dst_split

        print_colored(f"\n  📦 Split: {dst_split.upper()}", Colors.HEADER)

        image_files = scan_directory_with_animation(src_images, f"positive {dst_split} images")

        if image_files:
            print_colored(f"  🚀 Copying {len(image_files)} images with labels...", Colors.OKCYAN)

            # BARRE MULTICOLORE pour les positives
            for img_file in tqdm(
                image_files,
                desc=f"  {Colors.OKGREEN}✅ Positives{Colors.ENDC}",
                unit="img",
                bar_format='{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                colour='green',
                ncols=100
            ):
                shutil.copy2(img_file, dst_images / img_file.name)
                label_file = src_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, dst_labels / label_file.name)
                else:
                    (dst_labels / f"{img_file.stem}.txt").touch()
                total_positives += 1

            print_colored(f"  ✅ {len(image_files)} images copiées avec succès!", Colors.OKGREEN)

    # TRAITEMENT DES IMAGES NÉGATIVES (BACKGROUNDS)
    print_header("❌ ÉTAPE 4/5 - COPIE DES BACKGROUNDS (SANS FRAMBOISES)", "─")

    total_negatives = 0
    bg_images_dir = BACKGROUND_DIR / 'train' / "images"
    dst_images = TARGET_DIR / "images" / "train"

    print_colored("  🎯 Ces images N'ONT PAS de labels = vrais négatifs!", Colors.WARNING)
    print_colored("  💡 Elles vont réduire les faux positifs (mains, visages, objets rouges)", Colors.OKCYAN)
    print()

    bg_files = scan_directory_with_animation(bg_images_dir, "background images")

    if bg_files:
        print_colored(f"  🚀 Copying {len(bg_files)} negative images (NO LABELS)...", Colors.WARNING)

        # BARRE ROUGE pour les négatives
        for img_file in tqdm(
            bg_files,
            desc=f"  {Colors.FAIL}❌ Negatives{Colors.ENDC}",
            unit="img",
            bar_format='{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            colour='red',
            ncols=100
        ):
            dst_path = dst_images / f"neg_{img_file.name}"
            shutil.copy2(img_file, dst_path)
            total_negatives += 1

        print_colored(f"  ✅ {len(bg_files)} backgrounds copiés (SANS labels)!", Colors.OKGREEN)

    # VÉRIFICATION FINALE
    print_header("🔍 ÉTAPE 5/5 - VÉRIFICATION FINALE", "─")

    print_colored("  🔬 Scanning final dataset...", Colors.OKCYAN)
    animated_spinner(0.5)

    for split in ['train', 'val', 'test']:
        images_dir = TARGET_DIR / "images" / split
        labels_dir = TARGET_DIR / "labels" / split

        if not images_dir.exists():
            continue

        image_files = set([f.stem for f in images_dir.glob("*.jpg")] +
                         [f.stem for f in images_dir.glob("*.png")])
        label_files = set([f.stem for f in labels_dir.glob("*.txt")])
        images_without_labels = image_files - label_files

        print_colored(f"\n  📊 {split.upper()} Split:", Colors.HEADER)
        print_colored(f"     Total images: {len(image_files)}", Colors.OKBLUE)
        print(create_progress_bar(len(image_files), len(image_files), "     ", Colors.OKBLUE, 30))

        print_colored(f"     Images WITH labels: {len(label_files)}", Colors.OKGREEN)
        print(create_progress_bar(len(label_files), len(image_files), "     ", Colors.OKGREEN, 30))

        print_colored(f"     Images WITHOUT labels (negatives): {len(images_without_labels)}", Colors.FAIL)
        print(create_progress_bar(len(images_without_labels), len(image_files), "     ", Colors.FAIL, 30))

    # RÉSUMÉ FINAL ULTRA COLORÉ
    print_header("🎉 PRÉPARATION TERMINÉE AVEC SUCCÈS ! 🎉", "━")

    print_colored("📊 RÉSUMÉ GLOBAL:", Colors.HEADER)
    print()
    print_colored(f"  ✅ Images positives (avec labels):  {total_positives}", Colors.OKGREEN)
    print("  " + "█" * min(60, total_positives // 8))
    print()
    print_colored(f"  ❌ Images négatives (SANS labels):  {total_negatives}", Colors.FAIL)
    print("  " + "█" * min(60, total_negatives // 11))
    print()
    print_colored(f"  📈 TOTAL images d'entraînement:     {total_positives + total_negatives}", Colors.OKCYAN)
    print("  " + "█" * 60)
    print()

    # RATIO
    if total_positives + total_negatives > 0:
        ratio = (total_negatives / (total_positives + total_negatives)) * 100
        print_colored(f"  📊 Ratio négatifs: {ratio:.1f}%", Colors.WARNING)

    print()
    print_header("💡 PROCHAINE ÉTAPE", "─")

    print_colored("  🚀 Pour lancer l'entraînement:", Colors.OKCYAN)
    print()
    print_colored("     MODE RAPIDE (30 epochs, 1-2h):", Colors.OKGREEN)
    print_colored("     → python3 train_with_negatives.py --mode fast", Colors.BOLD)
    print()
    print_colored("     MODE COMPLET (120 epochs, 5-8h):", Colors.WARNING)
    print_colored("     → python3 train_with_negatives.py --mode full", Colors.BOLD)
    print()
    print_colored("     OU SCRIPT AUTOMATIQUE:", Colors.HEADER)
    print_colored("     → ./launch_training.sh fast", Colors.BOLD)
    print()

    print_header("✨ DATASET PRÊT POUR L'ENTRAÎNEMENT ! ✨", "━")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Arrêt demandé par l'utilisateur", Colors.WARNING)
    except Exception as e:
        print_colored(f"\n❌ ERREUR: {e}", Colors.FAIL)
        raise
