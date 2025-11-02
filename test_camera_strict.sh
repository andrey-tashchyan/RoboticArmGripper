#!/bin/bash
# Mode STRICT - Validation maximale pour éviter les faux positifs

echo "🎯 MODE STRICT - Validation maximale (zéro faux positifs)"
echo ""

source .venv311/bin/activate
BEST_MODEL=$(ls -t runs/raspberry_detect/train_*/weights/best.pt 2>/dev/null | head -1)

python raspberry_cam.py \
    --src 0 \
    --model "$BEST_MODEL" \
    --device mps \
    --conf 0.50 \
    --strict \
    --sensitivity 1.0 \
    --auto-calib \
    --debug \
    --roi-w 0.5 \
    --roi-h 0.5 \
    --min-frames 5 \
    --max-skin-ratio 0.15

echo "✅ Mode strict terminé"
