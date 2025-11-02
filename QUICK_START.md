# 🚀 Guide de démarrage ultra-rapide

## En 3 étapes simples

### 1️⃣ Préparer le dataset

```bash
python3 prepare_dataset_with_negatives.py
```

Sortie :
```
🍓 PREPARING YOLOV8 DATASET WITH NEGATIVE SAMPLES 🍓
  Positives: 100%|████████████| 499/499
  Negatives: 100%|████████████| 669/669
✅ DATASET PREPARATION COMPLETE ✅
```

---

### 2️⃣ Choisir votre mode

#### ⚡ Mode RAPIDE (recommandé pour commencer)

```bash
source .venv311/bin/activate
python3 train_with_negatives.py --mode fast
```

- ⏱️ 30 epochs (~1-2 heures)
- 🎯 Pour tester rapidement

#### 🏋️ Mode COMPLET (production)

```bash
source .venv311/bin/activate
python3 train_with_negatives.py --mode full
```

- ⏱️ 120 epochs (~5-8 heures)
- 🎯 Pour le modèle final

---

### 3️⃣ Ou utiliser le script automatique

```bash
./launch_training.sh fast    # Mode rapide
# ou
./launch_training.sh full    # Mode complet
```

---

## 📊 Pendant l'entraînement

Le script affiche :
- 📈 Statistiques du dataset
- 🔄 Barres de progression
- 📊 Métriques en temps réel

```
🍓 YOLOV8 RASPBERRY DETECTION - TRAINING WITH NEGATIVES 🍓

📊 Dataset Statistics
  Total training images: 1,168
  ✅ Images with labels (positives): 499
  ❌ Images without labels (negatives): 669
  📈 Negative ratio: 57.3%

💡 Loaded 669 negative background images with no labels.

🚀 STARTING TRAINING - FAST MODE 🚀
```

---

## 💾 Résultats

Après l'entraînement, vos modèles sont dans :

```
runs/raspberry_detect/train_fast/weights/best.pt     # Mode rapide
runs/raspberry_detect/train_full/weights/best.pt     # Mode complet
```

---

## 🎯 Utiliser le modèle

```bash
# Prédiction sur une image
yolo predict model=runs/raspberry_detect/train_fast/weights/best.pt source=image.jpg

# Prédiction sur une vidéo
yolo predict model=runs/raspberry_detect/train_fast/weights/best.pt source=video.mp4
```

---

## 🔧 Comparaison des modes

| Aspect | Mode RAPIDE | Mode COMPLET |
|--------|-------------|--------------|
| **Epochs** | 30 | 120 |
| **Durée** | 1-2h | 5-8h |
| **Patience** | 10 | 30 |
| **Usage** | Test, validation | Production |
| **Qualité** | Bonne | Excellente |

---

## ❓ En cas de problème

### "MPS out of memory"
Le batch size est déjà optimisé (8). Si ça persiste, éditez `train_with_negatives.py` ligne 23/29 et mettez `'batch': 4`

### Dataset non préparé
Relancez : `python3 prepare_dataset_with_negatives.py`

### Images négatives non détectées
Vérifiez :
```bash
ls data/raspberries/images/train/ | wc -l    # Doit être 1168
ls data/raspberries/labels/train/ | wc -l    # Doit être 499
```

---

## 📚 Documentation complète

Pour plus de détails : [README_TRAINING.md](README_TRAINING.md)

---

**C'est tout ! Bonne chance avec votre entraînement ! 🍓🚀**
