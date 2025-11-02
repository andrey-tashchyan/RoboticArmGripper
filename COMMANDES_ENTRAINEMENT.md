# 🚀 COMMANDES POUR LANCER L'ENTRAÎNEMENT

## 📋 Résumé Ultra-Rapide

### ⚡ ÉTAPE 1 : Préparer le dataset (une seule fois)

```bash
# Version standard
python3 prepare_dataset_with_negatives.py

# Version avec MAXIMUM de barres colorées 🌈
python3 prepare_dataset_colorful.py
```

---

### 🏃 ÉTAPE 2 : Lancer l'entraînement

#### 🟢 MODE RAPIDE (30 epochs, 1-2 heures)

```bash
# Option 1 : Python direct
source .venv311/bin/activate
python3 train_with_negatives.py --mode fast

# Option 2 : Script automatique
./launch_training.sh fast
```

#### 🔴 MODE COMPLET (120 epochs, 5-8 heures)

```bash
# Option 1 : Python direct
source .venv311/bin/activate
python3 train_with_negatives.py --mode full

# Option 2 : Script automatique
./launch_training.sh full
```

---

## 📊 Différences entre les modes

| Aspect | MODE FAST | MODE FULL |
|--------|-----------|-----------|
| **Epochs** | 30 | 120 |
| **Durée (avec GPU)** | 1-2h | 5-8h |
| **Durée (sans GPU)** | 4-6h | 15-20h |
| **Patience** | 10 epochs | 30 epochs |
| **Usage** | ✅ Tests, validation | ✅ Production |
| **Qualité** | Bonne | Excellente |

---

## 🎯 Workflow Complet

### Première utilisation

```bash
# 1. Activer l'environnement
source .venv311/bin/activate

# 2. Préparer le dataset (COLORÉ!)
python3 prepare_dataset_colorful.py

# 3. Test rapide
python3 train_with_negatives.py --mode fast

# 4. Si satisfait, production
python3 train_with_negatives.py --mode full
```

---

## 🌈 Version avec MAXIMUM de barres

### Script de préparation ultra-visuel

```bash
python3 prepare_dataset_colorful.py
```

**Affiche :**
- 🔍 Barres de scan des dossiers
- 🏗️ Barres de création des dossiers
- 🍓 Barres VERTES pour images positives
- ❌ Barres ROUGES pour images négatives (backgrounds)
- 📊 Barres de vérification finale
- 🎨 Résumé coloré avec statistiques

---

## 💻 Commandes par Système

### macOS / Linux

```bash
# Activer environnement
source .venv311/bin/activate

# Préparer (coloré)
python3 prepare_dataset_colorful.py

# Entraîner (rapide)
python3 train_with_negatives.py --mode fast

# OU script auto
./launch_training.sh fast
```

### Windows

```cmd
:: Activer environnement
venv\Scripts\activate.bat

:: Préparer (coloré)
python prepare_dataset_colorful.py

:: Entraîner (rapide)
python train_with_negatives.py --mode fast

:: OU script auto
launch_training.bat fast
```

---

## 📈 Ce qui se passe pendant l'entraînement

```
🍓 YOLOV8 RASPBERRY DETECTION - TRAINING WITH NEGATIVES 🍓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  MODE: FAST - Mode rapide (30 epochs)

📊                            Dataset Statistics                              📊
────────────────────────────────────────────────────────────────────────────────
  Total training images: 1,168
  ✅ Images with labels (positives): 499
  ❌ Images without labels (negatives): 669
  📈 Negative ratio: 57.3%
────────────────────────────────────────────────────────────────────────────────

💡 Loaded 669 negative background images with no labels.
   These will be treated as empty scenes to reduce false positives.

📦 Loading model: yolov8s.pt
Model loading: 100%|████████████| 100/100

🚀 STARTING TRAINING - FAST MODE 🚀
```

---

## 🎮 Contrôles pendant l'entraînement

| Action | Commande |
|--------|----------|
| **Arrêter** | `Ctrl + C` |
| **Voir logs** | `tail -f runs/raspberry_detect/train_fast/results.csv` |
| **Pause** | Impossible (arrêter et reprendre plus tard) |

---

## 📁 Résultats après entraînement

```
runs/raspberry_detect/train_fast/    (ou train_full/)
├── weights/
│   ├── best.pt         ← MEILLEUR MODÈLE (utiliser celui-ci)
│   └── last.pt         ← Dernier epoch
├── results.png         ← Courbes d'entraînement
├── confusion_matrix.png
└── results.csv         ← Métriques par epoch
```

---

## 🎯 Utiliser le modèle entraîné

### Prédiction sur une image

```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=mon_image.jpg \
  conf=0.5
```

### Prédiction sur une vidéo

```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=ma_video.mp4 \
  conf=0.5
```

### Webcam en temps réel

```bash
yolo predict \
  model=runs/raspberry_detect/train_fast/weights/best.pt \
  source=0 \
  show=True
```

---

## 🔥 Tips & Astuces

### Accélérer l'entraînement
```bash
# Réduire la taille des images (plus rapide, moins précis)
python3 train_with_negatives.py --mode fast
# Puis éditer train_with_negatives.py ligne 37: IMGSZ = 640
```

### Suivre en temps réel
```bash
# Dans un autre terminal
watch -n 5 "tail -10 runs/raspberry_detect/train_fast/results.csv"
```

### Comparer deux modèles
```bash
# Valider modèle 1
yolo val model=runs/raspberry_detect/train_fast/weights/best.pt data=data/raspberries.yaml

# Valider modèle 2
yolo val model=runs/raspberry_detect/train_full/weights/best.pt data=data/raspberries.yaml
```

---

## ⚡ COMMANDES À RETENIR

```bash
# PRÉPARATION (avec barres colorées)
python3 prepare_dataset_colorful.py

# ENTRAÎNEMENT RAPIDE (30 epochs)
python3 train_with_negatives.py --mode fast

# ENTRAÎNEMENT COMPLET (120 epochs)
python3 train_with_negatives.py --mode full

# PRÉDICTION
yolo predict model=runs/raspberry_detect/train_fast/weights/best.pt source=image.jpg
```

---

## 🆘 En cas de problème

### Erreur "No module named 'ultralytics'"
```bash
source .venv311/bin/activate
pip install ultralytics tqdm
```

### Entraînement très lent
Vous utilisez le CPU. Normal si pas de GPU.

### "Out of memory"
Éditez `train_with_negatives.py` lignes 23 et 29 :
```python
'batch': 4,  # Au lieu de 8
```

---

**🎉 Tout est prêt ! Lancez votre entraînement ! 🚀**
