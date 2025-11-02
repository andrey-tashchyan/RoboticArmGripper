#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cv_ripeness.py — Vision seule (lecture caméra + décision de maturité par couleur rouge)

Objectif
--------
Lire un flux caméra, estimer la "rougeur" (en HSV) d'une baie (framboise/fraise),
et décider si elle est mûre (score >= seuil) ou non (score < seuil).

Fonctionnalités
---------------
- Trois modes CLI : `calibrate`, `run`, `image`
- Calibration : sliders OpenCV pour ajuster les bornes HSV des deux bandes de rouge
- Exécution : calcul d'un score de rouge, comparaison à un seuil, option d'affichage
- Mode image : évaluer une photo fixe (jpg/png)

Dépendances
-----------
pip install opencv-python numpy

Exemples
--------
# 1) Calibrer les seuils HSV sous votre éclairage (sauvegarde config_hsv.json)
python cv_ripeness.py calibrate --camera-index 0

# 2) Lancer le test en "live" avec affichage
python cv_ripeness.py run --camera-index 0 --threshold 0.15 --show

# 3) Même chose mais sans afficher de fenêtre (console seulement)
python cv_ripeness.py run --camera-index 0 --threshold 0.15 --no-show

# 4) Tester sur une image fixe (avec affichage)
python cv_ripeness.py image --path chemin/vers/photo.jpg --threshold 0.15 --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────────────────────────
# Paramètres par défaut (modifiez-les au besoin)
# ────────────────────────────────────────────────────────────────────────────────

# Caméra
DEFAULT_CAMERA_INDEX = 0              # 0 = première caméra détectée
DEFAULT_FRAME_WIDTH = 640             # largeur demandée à la caméra
DEFAULT_FRAME_HEIGHT = 480            # hauteur demandée à la caméra

# Décision
DEFAULT_THRESHOLD = 0.15              # seuil de "proportion de rouge" pour dire "mûre"

# Fichier de configuration (optionnel) pour mémoriser les bornes HSV calibrées
CFG_PATH = Path("config_hsv.json")

# Bornes HSV par défaut pour la couleur ROUGE (H∈[0,180], S,V∈[0,255] en OpenCV)
# Le rouge "entoure" 0° → on utilise deux bandes : [0..10] et [170..180]
DEFAULT_HSV = {
    "low1":  [  0, 100,  60],   # bande basse   (rouge autour de 0°)
    "high1": [ 10, 255, 255],
    "low2":  [170, 100,  60],   # bande haute   (rouge autour de 180°)
    "high2": [180, 255, 255],
}


# ────────────────────────────────────────────────────────────────────────────────
# Utilitaires de configuration
# ────────────────────────────────────────────────────────────────────────────────

def load_hsv_config(cfg_path: Path = CFG_PATH) -> Dict[str, list]:
    """
    Charge un dictionnaire de bornes HSV depuis config_hsv.json, si présent.
    Sinon, retourne les bornes par défaut.

    Format attendu :
    {
        "low1": [H,S,V],
        "high1":[H,S,V],
        "low2": [H,S,V],
        "high2":[H,S,V]
    }
    """
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
            for k in ("low1", "high1", "low2", "high2"):
                if k not in data or not isinstance(data[k], list) or len(data[k]) != 3:
                    raise ValueError(f"Clé '{k}' absente ou mal formée.")
            return data
        except Exception as e:
            print(f"[WARN] Impossible de lire {cfg_path}: {e}. On utilise les valeurs par défaut.")
    return DEFAULT_HSV.copy()


def save_hsv_config(hsv_cfg: Dict[str, list], cfg_path: Path = CFG_PATH) -> None:
    """Sauvegarde les bornes HSV dans un fichier JSON (lisible et versionnable)."""
    cfg_path.write_text(json.dumps(hsv_cfg, indent=2))
    print(f"[INFO] Paramètres HSV sauvegardés dans {cfg_path.resolve()}")


# ────────────────────────────────────────────────────────────────────────────────
# Caméra
# ────────────────────────────────────────────────────────────────────────────────

def ensure_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Ouvre la caméra et tente de régler la résolution.
    Lève une exception si la caméra ne renvoie pas d'image.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            "Impossible d'ouvrir la caméra ou de lire un premier frame. "
            "Vérifie l'index caméra (--camera-index) et les permissions."
        )
    return cap


# ────────────────────────────────────────────────────────────────────────────────
# Vision (calcul du masque rouge et du score)
# ────────────────────────────────────────────────────────────────────────────────

def red_mask_hsv(frame_bgr: np.ndarray, hsv_cfg: Dict[str, list]) -> np.ndarray:
    """
    Calcule un masque binaire des pixels 'rouges' en utilisant deux bandes HSV.
    - Conversion BGR -> HSV
    - Seuils min/max pour chaque bande
    - Union (OR) des deux masques
    - Petite morphologie (open + close) pour nettoyer le bruit
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    l1 = np.array(hsv_cfg["low1"],  dtype=np.uint8)
    h1 = np.array(hsv_cfg["high1"], dtype=np.uint8)
    l2 = np.array(hsv_cfg["low2"],  dtype=np.uint8)
    h2 = np.array(hsv_cfg["high2"], dtype=np.uint8)

    m1 = cv2.inRange(hsv, l1, h1)
    m2 = cv2.inRange(hsv, l2, h2)

    mask = cv2.bitwise_or(m1, m2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def ripeness_score(mask: np.ndarray) -> float:
    """Proportion de pixels 'rouges' (masque binaire) ∈ [0,1]."""
    return float(mask.mean() / 255.0)


# ────────────────────────────────────────────────────────────────────────────────
# CALIBRATION (sliders)
# ────────────────────────────────────────────────────────────────────────────────

def run_calibration(camera_index: int, width: int, height: int) -> None:
    """
    Ouvre une fenêtre avec 6 sliders pour régler les 2 bandes HSV (low/high).
    Affiche côte-à-côte :
      - l'image d'origine
      - le masque binaire (blanc = rouge détecté)
    Commandes :
      - 's' : sauvegarder les bornes dans config_hsv.json
      - 'q' : quitter
    """
    hsv_cfg = load_hsv_config()
    cap = ensure_camera(camera_index, width, height)

    win = "calibrate (s=save, q=quit)"
    cv2.namedWindow(win)

    def add_slider(name: str, value: int, maxv: int = 255) -> None:
        cv2.createTrackbar(name, win, value, maxv, lambda _x: None)

    # Sliders bande 1 (autour de 0°)
    add_slider("low1_H",  hsv_cfg["low1"][0], 180)
    add_slider("low1_S",  hsv_cfg["low1"][1], 255)
    add_slider("low1_V",  hsv_cfg["low1"][2], 255)
    add_slider("high1_H", hsv_cfg["high1"][0], 180)
    add_slider("high1_S", hsv_cfg["high1"][1], 255)
    add_slider("high1_V", hsv_cfg["high1"][2], 255)

    # Sliders bande 2 (autour de 180°)
    add_slider("low2_H",  hsv_cfg["low2"][0], 180)
    add_slider("low2_S",  hsv_cfg["low2"][1], 255)
    add_slider("low2_V",  hsv_cfg["low2"][2], 255)
    add_slider("high2_H", hsv_cfg["high2"][0], 180)
    add_slider("high2_S", hsv_cfg["high2"][1], 255)
    add_slider("high2_V", hsv_cfg["high2"][2], 255)

    print("[Calib] Ajuste les sliders. 's' pour sauver, 'q' pour quitter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERREUR] Lecture caméra impossible.")
            break

        # Lire positions des sliders
        hsv_cfg["low1"]  = [cv2.getTrackbarPos("low1_H",  win),
                            cv2.getTrackbarPos("low1_S",  win),
                            cv2.getTrackbarPos("low1_V",  win)]
        hsv_cfg["high1"] = [cv2.getTrackbarPos("high1_H", win),
                            cv2.getTrackbarPos("high1_S", win),
                            cv2.getTrackbarPos("high1_V", win)]
        hsv_cfg["low2"]  = [cv2.getTrackbarPos("low2_H",  win),
                            cv2.getTrackbarPos("low2_S",  win),
                            cv2.getTrackbarPos("low2_V",  win)]
        hsv_cfg["high2"] = [cv2.getTrackbarPos("high2_H", win),
                            cv2.getTrackbarPos("high2_S", win),
                            cv2.getTrackbarPos("high2_V", win)]

        mask = red_mask_hsv(frame, hsv_cfg)
        score = ripeness_score(mask)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([frame, mask_bgr])

        cv2.putText(vis, f"score={score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_hsv_config(hsv_cfg)

    cap.release()
    cv2.destroyAllWindows()


# ────────────────────────────────────────────────────────────────────────────────
# RUN (décision mûre / pas mûre, caméra)
# ────────────────────────────────────────────────────────────────────────────────

def run_loop(camera_index: int,
             width: int,
             height: int,
             threshold: float,
             show: bool = True) -> None:
    """
    Boucle caméra : calcule en continu le score "rouge" et la décision.
    """
    hsv_cfg = load_hsv_config()
    cap = ensure_camera(camera_index, width, height)

    print("[INFO] Appuie sur 'q' dans la fenêtre pour quitter (ou Ctrl+C en console).")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERREUR] Lecture caméra impossible. Arrêt.")
            break

        mask = red_mask_hsv(frame, hsv_cfg)
        score = ripeness_score(mask)
        is_ripe = score >= threshold

        if show:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis = np.hstack([frame, mask_bgr])

            txt = f"score={score:.3f}  |  seuil={threshold:.3f}  |  decision={'MURE' if is_ripe else 'PAS MURE'}"
            color = (0, 200, 0) if is_ripe else (0, 0, 200)
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            cv2.imshow("ripeness (q pour quitter)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        else:
            print(f"score={score:.3f} → {'MURE' if is_ripe else 'PAS MURE'}")
            cv2.waitKey(200)  # petit délai pour ne pas spammer la console

    cap.release()
    if show:
        cv2.destroyAllWindows()


# ────────────────────────────────────────────────────────────────────────────────
# MODE IMAGE (photo fixe)
# ────────────────────────────────────────────────────────────────────────────────

def run_on_image(image_path: str, threshold: float, show: bool = True) -> None:
    """
    Analyse une image fixe (photo) au lieu du flux caméra.
    Affiche le score de rougeur et la décision mûre/pas mûre.
    """
    hsv_cfg = load_hsv_config()
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image: {image_path}")

    mask = red_mask_hsv(img, hsv_cfg)
    score = ripeness_score(mask)
    is_ripe = score >= threshold

    print(f"score={score:.3f}  seuil={threshold:.3f}  →  {'MURE' if is_ripe else 'PAS MURE'}")

    if show:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([img, mask_bgr])
        cv2.putText(vis, f"score={score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("image | mask (touche pour fermer)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ────────────────────────────────────────────────────────────────────────────────
# Interface CLI (argparse)
# ────────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vision seule : lecture caméra + masque rouge HSV + décision mûre/pas mûre."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Sous-commande : calibrate
    sp_cal = sub.add_parser("calibrate", help="Mode calibration (sliders HSV, 's' pour sauver).")
    sp_cal.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    sp_cal.add_argument("--frame-width",  type=int, default=DEFAULT_FRAME_WIDTH)
    sp_cal.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)

    # Sous-commande : run (caméra)
    sp_run = sub.add_parser("run", help="Mode exécution live (caméra).")
    sp_run.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    sp_run.add_argument("--frame-width",  type=int, default=DEFAULT_FRAME_WIDTH)
    sp_run.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    sp_run.add_argument("--threshold",    type=float, default=DEFAULT_THRESHOLD,
                        help="Seuil de proportion de pixels rouges (0..1).")
    group = sp_run.add_mutually_exclusive_group()
    group.add_argument("--show",    action="store_true",  help="Afficher la fenêtre vidéo (défaut).")
    group.add_argument("--no-show", action="store_true",  help="Pas d'affichage (console uniquement).")

    # Sous-commande : image (photo fixe)
    sp_img = sub.add_parser("image", help="Évalue une image fixe (jpg/png).")
    sp_img.add_argument("--path", required=True, help="Chemin de l'image (jpg/png).")
    sp_img.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Seuil de décision (0..1).")
    group2 = sp_img.add_mutually_exclusive_group()
    group2.add_argument("--show", action="store_true", help="Afficher image et masque.")
    group2.add_argument("--no-show", action="store_true", help="Console seulement.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "calibrate":
        run_calibration(
            camera_index=args.camera_index,
            width=args.frame_width,
            height=args.frame_height
        )
        return

    if args.cmd == "run":
        show_flag = not args.no_show  # --no-show a priorité si fourni
        run_loop(
            camera_index=args.camera_index,
            width=args.frame_width,
            height=args.frame_height,
            threshold=args.threshold,
            show=show_flag
        )
        return

    if args.cmd == "image":
        show_flag = not args.no_show
        run_on_image(args.path, threshold=args.threshold, show=show_flag)
        return


if __name__ == "__main__":
    main()