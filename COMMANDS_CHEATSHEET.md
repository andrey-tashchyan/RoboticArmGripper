# 📝 Aide-mémoire des commandes

## 🏗️ Préparation

### Préparer le dataset avec barres de progression
```bash
python3 prepare_dataset_with_negatives.py
```

---

## 🚀 Entraînement

### Option 1 : Script Python direct

#### Mode RAPIDE (30 epochs)
```bash
source .venv311/bin/activate
python3 train_with_negatives.py --mode fast
```

#### Mode COMPLET (120 epochs)
```bash
source .venv311/bin/activate
python3 train_with_negatives.py --mode full
```

### Option 2 : Script Shell automatique

#### Mode RAPIDE
```bash
./launch_training.sh fast
```

#### Mode COMPLET
```bash
./launch_training.sh full
```

### Option 3 : Commande YOLO CLI manuelle

```bash
source .venv311/bin/activate

yolo detect train \
  model=yolov8s.pt \
  data=data/raspberries.yaml \
  epochs=30 \
  imgsz=896 \
  batch=8 \
  device=mps \
  project=runs/raspberry_detect \
  name=train_manual \
  hsv_h=0.015 hsv_s=0.6 hsv_v=0.45 \
  degrees=7 translate=0.10 scale=0.35 \
  shear=3 perspective=0.0005 \
  mosaic=0.6 mixup=0.15 copy_paste=0.4 erasing=0.25 \
  box=7 cls=0.6 dfl=1.5 \
  lr0=0.004 lrf=0.1 optimizer=AdamW \
  cos_lr=True warmup_epochs=3 patience=10
```

---

## 📊 Monitoring

### Voir les logs en temps réel
```bash
# Mode RAPIDE
tail -f runs/raspberry_detect/train_fast/results.csv

# Mode COMPLET
tail -f runs/raspberry_detect/train_full/results.csv
```

### Voir les métriques mises à jour
```bash
watch -n 5 "tail -20 runs/raspberry_detect/train_fast/results.csv"
```

### Vérifier le processus en cours
```bash
ps aux | grep train_with_negatives
```

---

## 🛑 Contrôle

### Arrêter l'entraînement
```bash
# Trouver le PID
ps aux | grep train_with_negatives

# Arrêter proprement (remplacer PID)
kill -SIGINT <PID>

# Ou forcer l'arrêt
pkill -f train_with_negatives
```

### Reprendre un entraînement interrompu
```bash
yolo detect train \
  model=runs/raspberry_detect/train_fast/weights/last.pt \
  data=data/raspberries.yaml \
  resume=True
```

---

## 🔍 Validation et Test

### Valider le modèle sur le set de validation
```bash
yolo val \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  data=data/raspberries.yaml \
  split=val
```

### Valider sur le set de test
```bash
yolo val \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  data=data/raspberries.yaml \
  split=test
```

---

## 🎯 Inférence

### Image unique
```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=path/to/image.jpg \
  conf=0.5 \
  save=True
```

### Dossier d'images
```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=path/to/images/ \
  conf=0.5 \
  save=True
```

### Vidéo
```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=path/to/video.mp4 \
  conf=0.5 \
  save=True
```

### Webcam (device 0)
```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=0 \
  conf=0.5 \
  show=True
```

---

## 📈 Visualisation

### Ouvrir les courbes d'entraînement
```bash
open runs/raspberry_detect/train_fast/results.png
```

### Ouvrir la matrice de confusion
```bash
open runs/raspberry_detect/train_fast/confusion_matrix.png
```

### Voir les prédictions sur validation
```bash
open runs/raspberry_detect/train_fast/val_batch0_pred.jpg
open runs/raspberry_detect/train_fast/val_batch1_pred.jpg
```

---

## 🗂️ Gestion des fichiers

### Lister les entraînements
```bash
ls -lh runs/raspberry_detect/
```

### Comparer deux modèles
```bash
# Modèle 1
yolo val model=runs/raspberry_detect/train_fast/weights/best.pt data=data/raspberries.yaml

# Modèle 2
yolo val model=runs/raspberry_detect/train_full/weights/best.pt data=data/raspberries.yaml
```

### Nettoyer les anciens runs
```bash
# Attention : supprime TOUS les runs
rm -rf runs/raspberry_detect/train_*

# Ou supprimer un run spécifique
rm -rf runs/raspberry_detect/train_fast
```

### Sauvegarder un modèle important
```bash
# Créer un dossier de sauvegarde
mkdir -p saved_models

# Copier le meilleur modèle avec un nom explicite
cp runs/raspberry_detect/train_full/weights/best.pt \
   saved_models/raspberry_v1_full_120epochs.pt
```

---

## 🔧 Vérifications

### Vérifier le dataset
```bash
# Nombre d'images d'entraînement
ls data/raspberries/images/train/ | wc -l    # Doit être 1168

# Nombre de labels d'entraînement
ls data/raspberries/labels/train/ | wc -l    # Doit être 499

# Nombre d'images négatives
ls data/raspberries/images/train/neg_* | wc -l    # Doit être 669
```

### Vérifier la configuration
```bash
cat data/raspberries.yaml
```

### Vérifier l'environnement
```bash
source .venv311/bin/activate
python3 -c "from ultralytics import YOLO; import torch; print(f'YOLO OK - PyTorch: {torch.__version__}')"
```

---

## 📦 Export du modèle

### Export en ONNX
```bash
yolo export \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  format=onnx \
  imgsz=896
```

### Export en TorchScript
```bash
yolo export \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  format=torchscript \
  imgsz=896
```

### Export en CoreML (pour iOS)
```bash
yolo export \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  format=coreml \
  imgsz=896
```

---

## 🐛 Debug

### Verbose mode
```bash
python3 train_with_negatives.py --mode fast 2>&1 | tee training.log
```

### Vérifier les logs YOLO
```bash
cat runs/raspberry_detect/train_fast/train.log
```

### Afficher la configuration utilisée
```bash
cat runs/raspberry_detect/train_fast/args.yaml
```

---

## 📊 Statistiques rapides

### Afficher les métriques finales
```bash
tail -1 runs/raspberry_detect/train_fast/results.csv
```

### Extraire mAP50
```bash
tail -1 runs/raspberry_detect/train_fast/results.csv | awk -F',' '{print $10}'
```

### Extraire Precision
```bash
tail -1 runs/raspberry_detect/train_fast/results.csv | awk -F',' '{print $7}'
```

---

## 🎓 Aide

### Aide du script d'entraînement
```bash
python3 train_with_negatives.py --help
```

### Aide YOLO
```bash
yolo help
yolo train help
yolo predict help
```

### Version
```bash
yolo version
```

---

**💡 Conseil:** Sauvegardez cette page dans vos favoris pour un accès rapide !
