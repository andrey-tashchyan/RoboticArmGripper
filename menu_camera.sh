#!/bin/bash
# Menu interactif pour lancer la caméra avec différents modes

clear
echo "==================================================================="
echo "🍓 MENU DÉTECTION DE FRAMBOISES - CAMÉRA TEMPS RÉEL"
echo "==================================================================="
echo ""
echo "Choisissez un mode de détection:"
echo ""
echo "  1) 🎯 Mode ÉQUILIBRÉ (recommandé)"
echo "     → Bon compromis sensibilité/précision"
echo "     → Idéal pour usage quotidien"
echo ""
echo "  2) 🚀 Mode SIMPLE (permissif)"
echo "     → Toutes les détections visibles"
echo "     → Parfait pour démonstrations"
echo ""
echo "  3) 🔒 Mode STRICT (zéro faux positifs)"
echo "     → Validation maximale"
echo "     → Pour applications critiques"
echo ""
echo "  4) 📹 Analyser une VIDÉO"
echo "     → Traiter un fichier vidéo existant"
echo ""
echo "  5) 📊 Mode DEBUG (développement)"
echo "     → Métriques détaillées en temps réel"
echo ""
echo "  6) ❌ Quitter"
echo ""
echo "==================================================================="
echo -n "Votre choix [1-6]: "
read choice

case $choice in
    1)
        echo ""
        echo "🎯 Lancement du mode ÉQUILIBRÉ..."
        bash test_camera.sh
        ;;
    2)
        echo ""
        echo "🚀 Lancement du mode SIMPLE..."
        bash test_camera_simple.sh
        ;;
    3)
        echo ""
        echo "🔒 Lancement du mode STRICT..."
        bash test_camera_strict.sh
        ;;
    4)
        echo ""
        echo "📹 Analyse de vidéo"
        echo -n "Chemin du fichier vidéo: "
        read video_path

        if [ ! -f "$video_path" ]; then
            echo "❌ Fichier non trouvé: $video_path"
            exit 1
        fi

        source .venv311/bin/activate
        BEST_MODEL=$(ls -t runs/raspberry_detect/train_*/weights/best.pt 2>/dev/null | head -1)

        python raspberry_cam.py \
            --src "$video_path" \
            --model "$BEST_MODEL" \
            --device mps \
            --conf 0.40 \
            --strict \
            --sensitivity 1.5 \
            --save-vid "${video_path%.mp4}_analyzed.mp4"

        echo "✅ Vidéo analysée sauvegardée: ${video_path%.mp4}_analyzed.mp4"
        ;;
    5)
        echo ""
        echo "📊 Lancement du mode DEBUG..."
        source .venv311/bin/activate
        BEST_MODEL=$(ls -t runs/raspberry_detect/train_*/weights/best.pt 2>/dev/null | head -1)

        python raspberry_cam.py \
            --src 0 \
            --model "$BEST_MODEL" \
            --device mps \
            --conf 0.35 \
            --strict \
            --sensitivity 1.5 \
            --auto-calib \
            --debug \
            --save-log debug_session.csv

        echo "✅ Logs sauvegardés: debug_session.csv"
        ;;
    6)
        echo ""
        echo "👋 Au revoir!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Choix invalide. Veuillez choisir entre 1 et 6."
        exit 1
        ;;
esac

echo ""
echo "==================================================================="
echo "✅ Session terminée!"
echo "==================================================================="
