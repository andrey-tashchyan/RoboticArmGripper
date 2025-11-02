# 🍓 YOLOv8 Raspberry Detection - Entraînement avec Images Négatives

## 📋 Vue d'ensemble

Pipeline d'entraînement YOLOv8 optimisé avec **669 images négatives** pour réduire les faux positifs (mains, visages, objets rouges).

## 🚀 Guide de démarrage rapide

### 1️⃣ Préparer le dataset

```bash
python3 prepare_dataset_with_negatives.py
```

**Ce script va :**
- ✅ Copier 499 images positives avec leurs labels
- ❌ Copier 669 images négatives SANS labels
- 📊 Afficher des barres de progression pour chaque étape
- ✓ Vérifier l'intégrité du dataset

**Sortie attendue :**
```
🍓 PREPARING YOLOV8 DATASET WITH NEGATIVE SAMPLES 🍓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂 Processing TRAIN split
✅ Copying 499 positive samples (with labels)...
  Positives: 100%|████████████| 499/499 [00:03<00:00]
❌ Copying 669 negative samples (no labels)...
  Negatives: 100%|████████████| 669/669 [00:04<00:00]

✅ DATASET PREPARATION COMPLETE ✅
Total: 1,168 training images (499 positives + 669 negatives)
```

---

### 2️⃣ Lancer l'entraînement

#### 🏃 **Mode RAPIDE** (30 epochs - Test & Validation)

Pour tester rapidement la configuration :

```bash
source .venv311/bin/activate
python3 train_with_negatives.py --mode fast
```

**Caractéristiques :**
- ⏱️ **30 epochs** (≈ 1-2 heures)
- 🛑 **Patience : 10 epochs**
- 🎯 Idéal pour : tests, validation, debugging

---

#### 🏋️ **Mode COMPLET** (120 epochs - Production)

Pour l'entraînement final de production :

```bash
source .venv311/bin/activate
python3 train_with_negatives.py --mode full
```

**Caractéristiques :**
- ⏱️ **120 epochs** (≈ 5-8 heures)
- 🛑 **Patience : 30 epochs**
- 🎯 Idéal pour : modèle final, production

---

## 📊 Affichage pendant l'entraînement

Le script affiche :

```
🍓 YOLOV8 RASPBERRY DETECTION - TRAINING WITH NEGATIVES 🍓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  MODE: FAST - Mode rapide (30 epochs)

📊 Dataset Statistics
────────────────────────────────────────────────────────────────────────────────
  Total training images: 1,168
  ✅ Images with labels (positives): 499
  ❌ Images without labels (negatives): 669
  📈 Negative ratio: 57.3%

💡 Loaded 669 negative background images with no labels.
   These will be treated as empty scenes to reduce false positives.

📦 Loading model: yolov8s.pt
Model loading: 100%|████████████| 100/100

⚙️  Training Configuration
────────────────────────────────────────────────────────────────────────────────
  Model: yolov8s.pt
  Epochs: 30
  Batch size: 8
  Image size: 896x896
  Device: MPS
  Optimizer: AdamW
  Learning rate: 0.004 → 0.1
  Patience: 10 epochs

🚀 STARTING TRAINING - FAST MODE 🚀
```

---

## 📁 Structure du Dataset

```
data/raspberries/
├── images/
│   ├── train/          # 1,168 images
│   │   ├── Raspberries_100.jpg       # Images positives
│   │   ├── Raspberries_101.jpg
│   │   ├── neg_WIN_20220607_*.jpg    # Images négatives (préfixe 'neg_')
│   │   └── ...
│   ├── val/            # 99 images
│   └── test/           # 51 images
└── labels/
    ├── train/          # 499 fichiers .txt (SEULEMENT pour positives)
    ├── val/            # 99 fichiers .txt
    └── test/           # 51 fichiers .txt
```

**⚠️ Important :** Les images négatives n'ont PAS de fichiers .txt correspondants.

---

## 🎯 Résultats attendus

Après l'entraînement, vous devriez observer :

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| **Precision** | ~0.75 | ~0.85+ | ⬆️ +10-15% |
| **False Positives** | Élevé | Faible | ⬇️ -50-70% |
| **Recall** | ~0.80 | ~0.78-0.82 | ≈ Stable |

**Réduction des faux positifs sur :**
- ❌ Mains humaines
- ❌ Visages
- ❌ Objets rouges (vêtements, accessoires)
- ❌ Textures similaires

---

## 📈 Visualiser les résultats

### Pendant l'entraînement

```bash
# Voir les métriques en temps réel
watch -n 5 "tail -20 runs/raspberry_detect/train_fast/results.csv"

# Ou pour le mode full
watch -n 5 "tail -20 runs/raspberry_detect/train_full/results.csv"
```

### Après l'entraînement

```bash
# Ouvrir les courbes d'entraînement
open runs/raspberry_detect/train_fast/results.png

# Voir la matrice de confusion
open runs/raspberry_detect/train_fast/confusion_matrix.png

# Voir les prédictions sur validation
open runs/raspberry_detect/train_fast/val_batch0_pred.jpg
```

---

## 💾 Utiliser le modèle entraîné

### Inférence sur une image

```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=path/to/image.jpg \
  conf=0.5
```

### Inférence sur une vidéo

```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=path/to/video.mp4 \
  conf=0.5
```

### Validation sur le test set

```bash
yolo val \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  data=data/raspberries.yaml \
  split=test
```

---

## 🔧 Paramètres d'entraînement

### Modes disponibles

| Mode | Epochs | Patience | Durée | Usage |
|------|--------|----------|-------|-------|
| **fast** | 30 | 10 | 1-2h | Test rapide, validation |
| **full** | 120 | 30 | 5-8h | Production finale |

### Augmentation des données

```python
HSV:         h=0.015, s=0.6, v=0.45
Geometric:   degrees=7°, translate=0.1, scale=0.35, shear=3
Advanced:    mosaic=0.6, mixup=0.15, copy_paste=0.4, erasing=0.25
```

### Hyperparamètres

```python
Optimizer:   AdamW
LR:          0.004 → 0.1 (cosine)
Loss weights: box=7, cls=0.6, dfl=1.5
Batch size:  8 (optimisé pour MPS)
Image size:  896x896
```

---

## 🐛 Dépannage

### Erreur : "MPS backend out of memory"

**Solution :** Le batch size est déjà réduit à 8. Si l'erreur persiste :

```python
# Dans train_with_negatives.py, ligne 23 et 29
'batch': 4,  # Réduire de 8 à 4
```

### Les images négatives ne sont pas chargées

**Vérification :**

```bash
# Vérifier le nombre d'images sans labels
ls data/raspberries/images/train/ | wc -l    # Devrait être 1168
ls data/raspberries/labels/train/ | wc -l    # Devrait être 499
```

### Entraînement trop lent

**Solution :** Utiliser le mode `fast` ou réduire la taille d'image :

```python
# Dans train_with_negatives.py, ligne 37
IMGSZ = 640  # Au lieu de 896
```

---

## 📝 Fichiers générés

```
runs/raspberry_detect/train_fast/  (ou train_full/)
├── weights/
│   ├── best.pt         # Meilleur modèle (mAP max)
│   └── last.pt         # Dernier epoch
├── results.png         # Courbes d'entraînement
├── results.csv         # Métriques par epoch
├── confusion_matrix.png
├── val_batch0_pred.jpg # Prédictions sur validation
└── args.yaml           # Paramètres d'entraînement
```

---

## 🎓 Comprendre les images négatives

### Comment ça marche ?

1. **Découverte** : YOLO scanne `images/train/` et trouve toutes les images
2. **Lookup** : Pour chaque image, YOLO cherche un fichier `.txt` dans `labels/train/`
3. **Classification** :
   - ✅ Fichier existe → Image avec objets (ou vide si fichier vide)
   - ❌ Fichier manquant → **Vrai négatif** (scène vide)
4. **Apprentissage** : Le modèle apprend que ces scènes ne doivent produire AUCUNE détection

### Avantages

- ✅ Améliore la précision sans sacrifier le recall
- ✅ Réduit drastiquement les faux positifs
- ✅ Modèle plus robuste aux variations de scène
- ✅ Méthode standard dans la recherche (YOLO, Faster R-CNN, etc.)

---

## 📞 Support

En cas de problème :

1. Vérifier les logs : `cat runs/raspberry_detect/train_fast/results.csv`
2. Consulter : [TRAINING_SETUP_SUMMARY.md](TRAINING_SETUP_SUMMARY.md)
3. Issues GitHub : [anthropics/claude-code](https://github.com/anthropics/claude-code/issues)

---

## 📜 Licence

Projet académique - BA5 Production Development

---

**Créé avec ❤️ et Claude Code**
