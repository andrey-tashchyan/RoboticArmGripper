#!/bin/bash
# Script d'entraînement complet optimisé pour la détection de framboises
# Version avec barre de progression en temps réel

echo "==================================================================="
echo "🍓 ENTRAÎNEMENT COMPLET YOLOV8 - DÉTECTION DE FRAMBOISES"
echo "==================================================================="
echo ""
echo "Configuration:"
echo "  - Modèle: YOLOv8n (nano - rapide et léger)"
echo "  - Époques: 100"
echo "  - Taille d'image: 640"
echo "  - Batch size: 16"
echo "  - Device: MPS (Apple Silicon GPU)"
echo "  - Augmentations: Activées"
echo "  - Early stopping: 20 époques de patience"
echo "  - Barre de progression: ✅ Activée"
echo ""
echo "⏱️  Durée estimée: 30-60 minutes sur Apple Silicon M1/M2"
echo ""
read -p "Appuyez sur Entrée pour commencer l'entraînement..."

# Activer l'environnement virtuel
source .venv311/bin/activate

# Lancer l'entraînement avec le nouveau script
python train_raspberry_model_v2.py \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device mps \
    --patience 20 \
    --optimizer AdamW \
    --lr 0.001 \
    --weight-decay 0.0005

echo ""
echo "==================================================================="
echo "✅ ENTRAÎNEMENT TERMINÉ!"
echo "==================================================================="
echo ""
echo "📁 Le modèle entraîné se trouve dans: runs/raspberry_detect/train_*/weights/best.pt"
echo ""
echo "🚀 Pour tester avec raspberry_cam.py:"
echo "  source .venv311/bin/activate"
echo "  python raspberry_cam.py --src 0 --model runs/raspberry_detect/train_*/weights/best.pt --device mps --strict --sensitivity 1.5 --auto-calib --debug"
echo ""
