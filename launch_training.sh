#!/bin/bash

# 🍓 YOLOv8 Raspberry Detection - Script de lancement rapide
# Usage: ./launch_training.sh [fast|full]

set -e

# Couleurs pour le terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${PURPLE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🍓  YOLOv8 Raspberry Detection - Training with Negatives  🍓"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

# Vérifier l'argument
MODE="${1:-full}"

if [[ "$MODE" != "fast" && "$MODE" != "full" ]]; then
    echo -e "${RED}❌ Erreur: Mode invalide '$MODE'${NC}"
    echo -e "${YELLOW}Usage: $0 [fast|full]${NC}"
    echo ""
    echo "Modes disponibles:"
    echo "  ${GREEN}fast${NC} - Entraînement rapide (30 epochs, ~1-2h)"
    echo "  ${GREEN}full${NC} - Entraînement complet (120 epochs, ~5-8h)"
    exit 1
fi

# Afficher le mode sélectionné
if [[ "$MODE" == "fast" ]]; then
    echo -e "${CYAN}⚡ Mode RAPIDE sélectionné${NC}"
    echo "   • 30 epochs"
    echo "   • Patience: 10 epochs"
    echo "   • Durée estimée: 1-2 heures"
    echo "   • Idéal pour: tests et validation"
else
    echo -e "${GREEN}🏋️  Mode COMPLET sélectionné${NC}"
    echo "   • 120 epochs"
    echo "   • Patience: 30 epochs"
    echo "   • Durée estimée: 5-8 heures"
    echo "   • Idéal pour: production finale"
fi

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Vérifier l'environnement virtuel
if [[ ! -d ".venv311" ]]; then
    echo -e "${RED}❌ Erreur: Environnement virtuel '.venv311' introuvable${NC}"
    exit 1
fi

# Vérifier que le dataset existe
if [[ ! -d "data/raspberries/images/train" ]]; then
    echo -e "${YELLOW}⚠️  Dataset non préparé. Lancement de la préparation...${NC}"
    echo ""
    source .venv311/bin/activate
    python3 prepare_dataset_with_negatives.py
    echo ""
    echo -e "${GREEN}✅ Dataset préparé avec succès${NC}"
    echo ""
fi

# Demander confirmation
echo -e "${CYAN}📋 Prêt à démarrer l'entraînement en mode ${MODE}${NC}"
read -p "Continuer? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}❌ Entraînement annulé${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}🚀 Démarrage de l'entraînement...${NC}"
echo ""

# Activer l'environnement et lancer l'entraînement
source .venv311/bin/activate
python3 train_with_negatives.py --mode "$MODE"

# Résumé final
echo ""
echo -e "${PURPLE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Entraînement terminé!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

OUTPUT_DIR="runs/raspberry_detect/train_${MODE}"
echo -e "${GREEN}📂 Résultats sauvegardés dans: ${OUTPUT_DIR}/${NC}"
echo ""
echo "Fichiers générés:"
echo "  • ${OUTPUT_DIR}/weights/best.pt     (meilleur modèle)"
echo "  • ${OUTPUT_DIR}/weights/last.pt     (dernier epoch)"
echo "  • ${OUTPUT_DIR}/results.png         (courbes d'entraînement)"
echo "  • ${OUTPUT_DIR}/confusion_matrix.png"
echo ""
echo -e "${CYAN}💡 Pour utiliser le modèle:${NC}"
echo "  yolo predict model=${OUTPUT_DIR}/weights/best.pt source=image.jpg"
echo ""
