#!/bin/bash
# Script d'entraînement rapide pour tester (10 époques)
# Version avec barre de progression

echo "==================================================================="
echo "🚀 ENTRAÎNEMENT RAPIDE - 10 ÉPOQUES (TEST)"
echo "==================================================================="
echo ""
echo "⏱️  Durée estimée: 5-10 minutes"
echo ""

# Activer l'environnement virtuel
source .venv311/bin/activate

# Lancer l'entraînement avec le nouveau script
python train_raspberry_model_v2.py \
    --model yolov8n.pt \
    --epochs 10 \
    --imgsz 416 \
    --batch 16 \
    --device mps \
    --patience 5

echo ""
echo "==================================================================="
echo "✅ Test terminé! Pour un entraînement complet, utilisez:"
echo "  bash full_train.sh"
echo "==================================================================="
