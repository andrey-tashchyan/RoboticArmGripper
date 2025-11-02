# 🪟 Guide d'installation pour Windows

## ✅ Compatibilité

Les scripts Python sont **100% compatibles Windows** avec quelques ajustements mineurs.

---

## 🔧 Configuration requise

### Logiciels nécessaires

1. **Python 3.11+** : https://www.python.org/downloads/
   - ⚠️ Cocher "Add Python to PATH" pendant l'installation

2. **Git** (optionnel) : https://git-scm.com/download/win

3. **GPU NVIDIA** (optionnel mais recommandé) :
   - CUDA Toolkit 11.8+ : https://developer.nvidia.com/cuda-downloads
   - cuDNN : https://developer.nvidia.com/cudnn

---

## 📥 Installation

### 1. Créer l'environnement virtuel

```cmd
cd C:\Users\VotreNom\Desktop\BA5\ProdDev\Robotic_Arm

:: Créer l'environnement
python -m venv venv

:: Activer l'environnement
venv\Scripts\activate.bat

:: Installer les dépendances
pip install ultralytics tqdm pyyaml
```

### 2. Vérifier l'installation

```cmd
python -c "from ultralytics import YOLO; import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Sortie attendue :**
```
PyTorch: 2.x.x
CUDA available: True  (si vous avez un GPU NVIDIA)
```

---

## ⚙️ Modifications nécessaires

### 1. **Device dans `train_with_negatives.py`**

**Ligne 38**, remplacez :

```python
# macOS (MPS)
DEVICE = "mps"
```

Par :

```python
# Windows avec GPU NVIDIA
DEVICE = "cuda"  # ou "0" pour GPU principal

# OU Windows sans GPU (CPU seulement - LENT)
DEVICE = "cpu"
```

### 2. **Chemins dans `data/raspberries.yaml`**

**Utilisez des chemins Windows OU relatifs :**

```yaml
# Option 1: Chemin absolu Windows (avec / ou \\)
path: C:/Users/VotreNom/Desktop/BA5/ProdDev/Robotic_Arm/data/raspberries

# Option 2: Chemin relatif (RECOMMANDÉ - fonctionne partout)
path: ./data/raspberries
```

---

## 🚀 Utilisation sur Windows

### Méthode 1 : Script Batch automatique

```cmd
:: Mode rapide (30 epochs)
launch_training.bat fast

:: Mode complet (120 epochs)
launch_training.bat full
```

### Méthode 2 : Commandes Python directes

```cmd
:: Activer l'environnement
venv\Scripts\activate.bat

:: Préparer le dataset (une seule fois)
python prepare_dataset_with_negatives.py

:: Lancer l'entraînement
python train_with_negatives.py --mode fast
:: OU
python train_with_negatives.py --mode full
```

### Méthode 3 : Version universelle (recommandée)

```cmd
venv\Scripts\activate.bat
python prepare_dataset_universal.py
python train_with_negatives.py --mode fast
```

---

## 📊 Différences Windows vs macOS

| Aspect | Windows | macOS/Linux |
|--------|---------|-------------|
| **Environnement virtuel** | `venv\Scripts\activate.bat` | `source .venv311/bin/activate` |
| **Device GPU** | `cuda` (NVIDIA) | `mps` (Apple Silicon) |
| **Séparateur de chemin** | `\` ou `/` | `/` |
| **Script shell** | `.bat` | `.sh` |
| **Émojis** | ✅ Supportés (Windows 10+) | ✅ Supportés |

---

## 🎯 Commandes Windows complètes

### Workflow complet

```cmd
:: 1. Ouvrir l'invite de commandes (cmd) ou PowerShell
cd C:\Users\VotreNom\Desktop\BA5\ProdDev\Robotic_Arm

:: 2. Activer l'environnement
venv\Scripts\activate.bat

:: 3. Préparer le dataset (une fois)
python prepare_dataset_with_negatives.py

:: 4. Vérifier que tout est OK
dir data\raspberries\images\train /b | find /c /v ""
dir data\raspberries\labels\train /b | find /c /v ""

:: Devrait afficher : 1168 et 499

:: 5. Tester rapidement (30 epochs, 1-2h)
python train_with_negatives.py --mode fast

:: 6. Si satisfait, production (120 epochs, 5-8h)
python train_with_negatives.py --mode full

:: 7. Utiliser le modèle
yolo predict model=runs\raspberry_detect\train_fast\weights\best.pt source=image.jpg
```

---

## 🔥 GPU NVIDIA : Configuration CUDA

### Vérifier CUDA

```cmd
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Installer PyTorch avec CUDA

Si CUDA n'est pas détecté, réinstallez PyTorch :

```cmd
:: Désinstaller PyTorch existant
pip uninstall torch torchvision

:: Réinstaller avec CUDA 11.8 (adapter selon votre version CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

:: Vérifier
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Performance GPU vs CPU

| Device | Temps (30 epochs) | Recommandation |
|--------|-------------------|----------------|
| **NVIDIA RTX 3060+** | 30-45 min | ✅ Idéal |
| **NVIDIA GTX 1660+** | 1-1.5h | ✅ Bon |
| **CPU Intel i7+** | 4-6h | ⚠️ Lent mais possible |
| **CPU Intel i5** | 6-10h | ❌ Très lent |

---

## 🐛 Dépannage Windows

### Erreur : "python n'est pas reconnu"

**Solution :** Python n'est pas dans le PATH

```cmd
:: Trouver Python
where python

:: Si rien, réinstaller Python et cocher "Add to PATH"
:: Ou ajouter manuellement : C:\Users\VotreNom\AppData\Local\Programs\Python\Python311
```

### Erreur : "No module named 'ultralytics'"

**Solution :** Environnement virtuel pas activé ou dépendances manquantes

```cmd
venv\Scripts\activate.bat
pip install -r requirements.txt
:: OU
pip install ultralytics tqdm pyyaml
```

### Erreur : "CUDA out of memory"

**Solution 1 :** Réduire le batch size

Dans `train_with_negatives.py`, lignes 23 et 29 :
```python
'batch': 4,  # Au lieu de 8
```

**Solution 2 :** Réduire la taille d'image

Ligne 37 :
```python
IMGSZ = 640  # Au lieu de 896
```

### Émojis ne s'affichent pas

**Solution :** Windows 10+ requis

```cmd
:: Dans cmd, exécuter :
chcp 65001

:: Puis relancer le script
```

---

## 📝 Fichier requirements.txt pour Windows

Créez `requirements.txt` :

```txt
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
pyyaml>=6.0
opencv-python>=4.7.0
pillow>=9.5.0
numpy>=1.24.0
```

Installation :
```cmd
pip install -r requirements.txt
```

---

## 🎨 PowerShell vs CMD

### PowerShell

```powershell
# Activer l'environnement (PowerShell)
.\venv\Scripts\Activate.ps1

# Si erreur "scripts désactivés"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CMD (Invite de commandes)

```cmd
:: Activer l'environnement (CMD)
venv\Scripts\activate.bat
```

---

## ✅ Checklist de compatibilité

- [ ] Python 3.11+ installé avec PATH
- [ ] Environnement virtuel créé (`python -m venv venv`)
- [ ] Dépendances installées (`pip install ultralytics tqdm pyyaml`)
- [ ] DEVICE modifié dans `train_with_negatives.py` (cuda ou cpu)
- [ ] Chemins relatifs dans `data/raspberries.yaml`
- [ ] CUDA vérifié si GPU NVIDIA (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Test du script : `python train_with_negatives.py --help`

---

## 🚀 Résumé pour Windows

**Installation :**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install ultralytics tqdm pyyaml
```

**Configuration :**
- Modifier `DEVICE = "cuda"` (ligne 38 de train_with_negatives.py)
- Utiliser chemins relatifs ou Windows (`C:/...`)

**Utilisation :**
```cmd
python prepare_dataset_with_negatives.py
python train_with_negatives.py --mode fast
```

**Ou script automatique :**
```cmd
launch_training.bat fast
```

---

## 📚 Ressources Windows

- [Python Windows](https://www.python.org/downloads/windows/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch Windows](https://pytorch.org/get-started/locally/)
- [Ultralytics Windows](https://docs.ultralytics.com/guides/windows/)

---

**Les scripts Python fonctionnent parfaitement sur Windows avec ces ajustements ! 🎉**
