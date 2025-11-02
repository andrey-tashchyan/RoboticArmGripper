#!/bin/bash
# Mode SIMPLE - Détection sans validation stricte (pour voir toutes les détections)

echo "🍓 MODE SIMPLE - Détection sans filtrage strict"
echo ""

source .venv311/bin/activate
BEST_MODEL=$(ls -t runs/raspberry_detect/train_*/weights/best.pt 2>/dev/null | head -1)

python raspberry_cam.py \
    --src 0 \
    --model "$BEST_MODEL" \
    --device mps \
    --conf 0.30 \
    --sensitivity 2.0 \
    --roi-w 0.8 \
    --roi-h 0.8 \
    --min-frames 1

echo "✅ Mode simple terminé"
