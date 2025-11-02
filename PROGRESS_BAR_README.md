# 🍓 Entraînement YOLOv8 avec Barre de Progression

## Nouveautés

Le nouveau script `train_raspberry_model_v2.py` inclut une **barre de progression en temps réel** pour suivre l'avancement de l'entraînement!

### ✨ Fonctionnalités

- 📊 **Barre de progression visuelle** avec tqdm
- ⏱️ **Temps écoulé et estimé** pour chaque époque
- 📈 **Métriques en temps réel**: loss, mAP50, mAP50-95
- 🎯 **Meilleur score** affiché en continu
- 🎨 **Emojis** pour une meilleure lisibilité
- ⚡ **Callbacks personnalisés** Ultralytics

## 🚀 Utilisation

### Entraînement Rapide (10 époques)

```bash
bash quick_train.sh
```

Ou directement:

```bash
source .venv311/bin/activate
python train_raspberry_model_v2.py \
    --model yolov8n.pt \
    --epochs 10 \
    --imgsz 416 \
    --batch 16 \
    --device mps \
    --patience 5
```

### Entraînement Complet (100 époques)

```bash
bash full_train.sh
```

Ou directement:

```bash
source .venv311/bin/activate
python train_raspberry_model_v2.py \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device mps \
    --patience 20
```

## 📊 Exemple d'Affichage

```
================================================================================
🍓 ENTRAÎNEMENT YOLOV8 - DÉTECTION DE FRAMBOISES
================================================================================
📦 Modèle de base: yolov8n.pt
📊 Dataset: Raspberry/data.yaml
🔄 Époques: 100
📐 Taille d'image: 640
📦 Batch size: 16
💻 Device: mps
⚙️  Optimiseur: AdamW
📈 Learning rate: 0.001
🎨 Augmentation: ✅ Activée
================================================================================

🔧 Chargement du modèle yolov8n.pt...
🚀 Démarrage de l'entraînement...
💾 Résultats: runs/raspberry_detect_20251102_123456

🍓 Entraînement: 100%|████████████| 100/100 [45:23<00:00, 27.2s/epoch] loss=0.8234 mAP50=0.912 best=0.923

================================================================================
✅ ENTRAÎNEMENT TERMINÉ!
================================================================================
⏱️  Durée totale: 45.4 minutes
🎯 Meilleur mAP50: 0.9234
================================================================================

================================================================================
📊 MÉTRIQUES FINALES
================================================================================
🎯 mAP50: 0.9234
🎯 mAP50-95: 0.7456
✓  Precision: 0.8912
✓  Recall: 0.8734
```

## 🎨 Informations Affichées

### Pendant l'Entraînement

- **Barre de progression** avec pourcentage
- **Nombre d'époques** (actuelle / totale)
- **Temps écoulé** pour chaque époque
- **Temps restant estimé**
- **Loss** (total = box + cls + dfl)
- **mAP50** après validation
- **Meilleur mAP50** depuis le début

### À la Fin

- ⏱️ **Durée totale** d'entraînement
- 🎯 **Métriques finales**: mAP50, mAP50-95, Precision, Recall
- 💾 **Chemins des fichiers** générés (best.pt, last.pt, results.csv)
- 🧪 **Résultats du test set**
- 🚀 **Commande prête à copier** pour utiliser le modèle

## 🔧 Options Avancées

### Tous les Arguments Disponibles

```bash
python train_raspberry_model_v2.py \
    --model yolov8n.pt \          # Modèle de base (n/s/m/l/x)
    --data Raspberry/data.yaml \  # Dataset YAML
    --epochs 100 \                # Nombre d'époques
    --imgsz 640 \                 # Taille d'image
    --batch 16 \                  # Batch size
    --device mps \                # Device (mps/cuda/cpu)
    --patience 20 \               # Early stopping
    --optimizer AdamW \           # Optimizer (AdamW/Adam/SGD)
    --lr 0.001 \                  # Learning rate
    --weight-decay 0.0005 \       # L2 regularization
    --save-dir runs/raspberry     # Dossier de sortie
```

### Désactiver les Augmentations

```bash
python train_raspberry_model_v2.py --epochs 50 --no-augment
```

### Utiliser un Modèle Plus Grand

```bash
python train_raspberry_model_v2.py --model yolov8s.pt --epochs 100 --batch 8
```

### Entraînement sur CPU

```bash
python train_raspberry_model_v2.py --device cpu --batch 4 --epochs 50
```

## 📈 Callbacks Personnalisés

Le script utilise les callbacks Ultralytics pour afficher les métriques:

- `on_train_epoch_start`: Initialise le timer d'époque
- `on_train_epoch_end`: Affiche loss et temps d'époque
- `on_fit_epoch_end`: Affiche mAP après validation
- `on_train_end`: Résumé final

## 🔄 Comparaison des Versions

| Fonctionnalité | v1 (original) | v2 (avec barre) |
|----------------|---------------|-----------------|
| Barre de progression | ❌ | ✅ |
| Temps restant estimé | ❌ | ✅ |
| Métriques en temps réel | ❌ | ✅ |
| Emojis | ❌ | ✅ |
| Callbacks personnalisés | ❌ | ✅ |
| Affichage propre | Verbose | Organisé |
| Meilleur score suivi | ❌ | ✅ |

## 🐛 Résolution de Problèmes

### La barre de progression ne s'affiche pas

Si `tqdm` n'est pas installé:

```bash
source .venv311/bin/activate
pip install tqdm
```

Le script fonctionnera quand même sans tqdm, mais sans la barre de progression visuelle.

### Conflits d'affichage

Si vous voyez des doublons d'affichage, c'est que le mode verbose d'Ultralytics interfère. Le script v2 désactive automatiquement `verbose=False`.

### Interruption de l'entraînement

Pour arrêter proprement l'entraînement:
- Appuyez sur `Ctrl+C`
- Le dernier checkpoint sera sauvegardé dans `last.pt`

## 💡 Conseils

1. **Utilisez toujours v2** pour un meilleur suivi
2. **Surveillez mAP50** - cible: >0.90
3. **Early stopping** s'active automatiquement si pas d'amélioration
4. **Sauvegarde automatique** tous les 10 époques
5. **Test final** sur le test set pour validation

## 📝 Fichiers Générés

```
runs/raspberry_detect/train_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt          ⭐ Utilisez celui-ci!
│   └── last.pt          (backup)
├── results.csv          📊 Toutes les métriques
├── results.png          📈 Graphiques de courbes
├── confusion_matrix.png 🔢 Matrice de confusion
├── F1_curve.png        📉 Courbe F1-score
├── PR_curve.png        📉 Precision-Recall
└── val_batch*.jpg      🖼️ Exemples de validation
```

---

**Bon entraînement! 🚀🍓**
