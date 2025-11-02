# 🪟 Démarrage rapide sur Windows

## ⚡ Installation en 5 minutes

### 1. Installer Python

1. Téléchargez Python 3.11+ : https://www.python.org/downloads/
2. **IMPORTANT** : Cochez "Add Python to PATH" pendant l'installation
3. Vérifiez :
   ```cmd
   python --version
   ```

### 2. Créer l'environnement virtuel

Ouvrez l'**invite de commandes** (cmd) ou **PowerShell** :

```cmd
cd C:\Users\VotreNom\Desktop\BA5\ProdDev\Robotic_Arm

python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements_full.txt
```

### 3. Vérifier l'installation

```cmd
python -c "from ultralytics import YOLO; print('✅ OK')"
python -c "import torch; print(f'Device: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 🚀 Lancer l'entraînement

### Option 1 : Script automatique (RECOMMANDÉ)

```cmd
launch_training.bat fast
```

### Option 2 : Commandes manuelles

```cmd
venv\Scripts\activate.bat
python prepare_dataset_with_negatives.py
python train_with_negatives.py --mode fast
```

---

## ✅ C'est tout !

**Tous les scripts sont maintenant 100% compatibles Windows** :

- ✅ Détection automatique du GPU (CUDA/CPU)
- ✅ Chemins relatifs (pas de modification nécessaire)
- ✅ Script .bat pour Windows
- ✅ Barres de progression fonctionnent
- ✅ Émojis supportés (Windows 10+)

---

## 🎯 Les 2 modes

### Mode RAPIDE (30 epochs, ~1-2h avec GPU)
```cmd
launch_training.bat fast
```

### Mode COMPLET (120 epochs, ~5-8h avec GPU)
```cmd
launch_training.bat full
```

---

## 🐛 Problèmes courants

### "python n'est pas reconnu"
Réinstallez Python et cochez "Add to PATH"

### Entraînement très lent
Vous utilisez le CPU. Pour accélérer :
1. Installez CUDA Toolkit : https://developer.nvidia.com/cuda-downloads
2. Réinstallez PyTorch avec CUDA :
   ```cmd
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### "Out of memory"
Modifiez `train_with_negatives.py` ligne 23 et 29 :
```python
'batch': 4,  # Au lieu de 8
```

---

## 📚 Documentation

- **Guide complet** : [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- **Démarrage rapide** : [QUICK_START.md](QUICK_START.md)
- **Commandes** : [COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md)

---

**🎉 Tout est prêt pour Windows !**
