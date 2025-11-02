#!/bin/bash
# Script pour tester la détection de framboises en temps réel avec la caméra

echo "==================================================================="
echo "🍓 TEST CAMÉRA - DÉTECTION DE FRAMBOISES EN TEMPS RÉEL"
echo "==================================================================="
echo ""

# Activer l'environnement virtuel
source .venv311/bin/activate

# Trouver le modèle le plus récent
BEST_MODEL=$(ls -t runs/raspberry_detect/train_*/weights/best.pt 2>/dev/null | head -1)

if [ -z "$BEST_MODEL" ]; then
    echo "❌ Aucun modèle entraîné trouvé!"
    echo "   Lancez d'abord: bash quick_train.sh ou bash full_train.sh"
    exit 1
fi

echo "✅ Modèle trouvé: $BEST_MODEL"
echo ""
echo "Configuration:"
echo "  - Mode: Détection temps réel"
echo "  - Device: MPS (Apple Silicon GPU)"
echo "  - Validation: Multi-cue stricte activée"
echo "  - Sensibilité: 1.5x (équilibré)"
echo "  - Calibration: Auto-calibration dynamique"
echo "  - Debug: Overlay activé"
echo ""
echo "📹 Ouverture de la caméra..."
echo ""
echo "Contrôles:"
echo "  - ESC ou 'q' : Quitter"
echo "  - La fenêtre affiche les détections en temps réel"
echo ""
echo "==================================================================="
echo ""

# Lancer la détection avec le modèle entraîné
python raspberry_cam.py \
    --src 0 \
    --model "$BEST_MODEL" \
    --device mps \
    --conf 0.40 \
    --strict \
    --sensitivity 1.5 \
    --auto-calib \
    --debug \
    --roi-w 0.6 \
    --roi-h 0.6 \
    --min-frames 3

echo ""
echo "==================================================================="
echo "✅ Session terminée!"
echo "==================================================================="
