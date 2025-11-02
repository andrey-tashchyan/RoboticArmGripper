#!/usr/bin/env python3
"""
Test de compatibilité Windows/macOS/Linux
Vérifie que tout est correctement installé et configuré.
"""

import sys
import platform
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_python_version():
    print("\n✓ Python version:")
    version = sys.version_info
    print(f"  {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("  ✅ OK (3.8+)")
        return True
    else:
        print("  ❌ ERREUR: Python 3.8+ requis")
        return False

def test_platform():
    print("\n✓ Plateforme:")
    os_name = platform.system()
    print(f"  {os_name} ({platform.platform()})")
    return True

def test_dependencies():
    print("\n✓ Dépendances:")

    dependencies = {
        'ultralytics': 'YOLO',
        'torch': 'PyTorch',
        'tqdm': 'Barres de progression',
        'yaml': 'Configuration YAML',
        'PIL': 'Pillow (images)',
        'cv2': 'OpenCV'
    }

    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name} ({module})")
        except ImportError:
            print(f"  ❌ {name} ({module}) - MANQUANT")
            all_ok = False

    return all_ok

def test_device():
    print("\n✓ Device disponible:")
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ NVIDIA GPU: {gpu_name}")
            print(f"     CUDA version: {torch.version.cuda}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print(f"  ✅ Apple Silicon (MPS)")
        else:
            device = "cpu"
            print(f"  ⚠️  CPU seulement (pas de GPU détecté)")
            print(f"     L'entraînement sera LENT")

        print(f"  → Device sélectionné: {device.upper()}")
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_paths():
    print("\n✓ Structure des fichiers:")

    required_paths = {
        'prepare_dataset_with_negatives.py': 'Script de préparation',
        'train_with_negatives.py': 'Script d\'entraînement',
        'data/raspberries.yaml': 'Configuration YOLO',
        'Raspberry': 'Dataset positif (dossier)',
        'background': 'Dataset négatif (dossier)'
    }

    all_ok = True
    for path, description in required_paths.items():
        p = Path(path)
        if p.exists():
            print(f"  ✅ {description}: {path}")
        else:
            print(f"  ⚠️  {description}: {path} - MANQUANT")
            if path in ['Raspberry', 'background']:
                all_ok = False  # Ces dossiers sont critiques

    return all_ok

def test_ultralytics():
    print("\n✓ Test Ultralytics:")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Petit modèle pour test
        print("  ✅ YOLO peut charger un modèle")
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    print_header("🧪 TEST DE COMPATIBILITÉ - YOLOv8 Raspberry Detection")

    print(f"\n📍 Dossier actuel: {Path.cwd()}")

    tests = [
        ("Python version", test_python_version),
        ("Plateforme", test_platform),
        ("Dépendances", test_dependencies),
        ("Device GPU/CPU", test_device),
        ("Structure fichiers", test_paths),
        ("Ultralytics", test_ultralytics)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erreur durant le test '{name}': {e}")
            results.append((name, False))

    # Résumé
    print_header("📊 RÉSUMÉ")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ OK" if result else "❌ ÉCHEC"
        print(f"  {status:12} {name}")

    print("\n" + "=" * 80)
    print(f"  Résultat: {passed}/{total} tests réussis")

    if passed == total:
        print("\n  🎉 TOUT EST OK ! Vous pouvez lancer l'entraînement.")
        print("\n  Commandes:")
        if platform.system() == "Windows":
            print("    launch_training.bat fast")
        else:
            print("    ./launch_training.sh fast")
        print("    python train_with_negatives.py --mode fast")
    else:
        print("\n  ⚠️  Certains tests ont échoué.")
        print("     Installez les dépendances manquantes:")
        print("     pip install -r requirements_full.txt")

    print("=" * 80)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
