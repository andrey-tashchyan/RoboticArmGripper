# -*- coding: utf-8 -*-
"""
vision_to_arduino.py — Vision → Arduino avec fond vert
-----------------------------------------------------
Ce script :
  1) Ouvre la **webcam intégrée** (index 0). Si indisponible, tente l’auto-détection.
  2) Détecte les pixels rouges (RR/GG/BB) ET retire le fond **vert** (masque vert soustrait).
  3) Prend la plus grande framboise détectée, évalue :
       - maturité (fraction de rouge dans la boîte englobante)
       - largeur max (double boucle lignes/colonnes)
       - largeur en mm via calibration pixels→mm (caméra à 10 cm)
  4) Si mûre, envoie à l'Arduino :  GRAB,<SMALL|LARGE>,<width_mm_int>
Affichage :
  - Fenêtres "camera" et "mask_red" (après retrait du vert)
Arrêts propres :
  - ESC ou 'q' ; fermeture de fenêtre ; Ctrl+C dans le terminal.
Raccourcis :
  - 'g' : forcer un GRAB (même sans détection)
  - 'v' : toggle affichage fenêtres
Dépendances (dans le venv) :
  pip install opencv-python numpy pyserial
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import serial
import serial.tools.list_ports as list_ports


# =========================
# Configuration
# =========================
@dataclass
class Config:
    # --- Caméra ---
    cam_index: int = 0        # 0 = webcam intégrée. Si échec, on bascule en auto-détection.
    cam_width: int = 1280
    cam_height: int = 720
    show_window: bool = True

    # --- Détection "rouge" (RR/GG/BB) ---
    # Un pixel est "rouge" si : R >= R_MIN  ET  R >= K_DOM*G  ET  R >= K_DOM*B
    R_MIN: int = 90
    K_DOM: float = 1.35

    # --- Masque "vert" (fond) à retirer ---
    # Un pixel est "vert" si : G >= G_MIN  ET  G >= K_G_DOM*R  ET  G >= K_G_DOM*B
    G_MIN: int = 60
    K_G_DOM: float = 1.25

    # --- Nettoyage masque ---
    morph_open_ks: int = 5
    morph_close_ks: int = 7
    min_area_px: int = 300

    # --- Maturité (fraction de rouge dans la boîte) ---
    red_area_frac_min: float = 0.40

    # --- Calibration (pixels → mm) à 10 cm ---
    REF_MM: float = 20.0        # largeur réelle d’un gabarit (mm)
    REF_PX: float = 64.0        # largeur mesurée en px à 10 cm
    override_mm_per_px: float = 0.0  # >0 pour forcer mm/px (sinon REF_MM/REF_PX)

    # --- Décision petite/grande (mm) ---
    small_large_threshold_mm: float = 25.0

    # --- Série (Arduino) ---
    baud: int = 115200
    send_when_ripe: bool = True
    verbose_serial: bool = True
    min_interval_s: float = 1.0  # anti-spam série


CFG = Config()


# =========================
# Caméra
# =========================
def autodetect_camera(max_index: int = 4) -> Optional[int]:
    """Retourne l'index de la première caméra dispo (0..max_index-1), sinon None."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None


def open_camera(cfg: Config):
    """
    Ouvre la webcam intégrée (index 0).
    Si échec (absence/occupée), essaye l’auto-détection.
    """
    def try_open(index: int):
        cap_ = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap_.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
        cap_.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
        return cap_ if cap_.isOpened() else None

    cap = try_open(cfg.cam_index)
    if cap is None:
        idx = autodetect_camera()
        if idx is None:
            print("⚠️  Aucune caméra disponible.")
            return None
        cap = try_open(idx)
        if cap is None:
            print(f"⚠️  Caméra index {idx} non disponible.")
            return None
        cfg.cam_index = idx

    print(f"📷 Caméra OK (index={cfg.cam_index})")
    return cap


# =========================
# Série (Arduino)
# =========================
def find_arduino_port() -> Optional[str]:
    """Repère un Arduino/clone par description/VID/PID. Retourne 'COMx' ou None."""
    for p in list_ports.comports():
        d = (p.description or "").upper()
        h = (p.hwid or "").upper()
        if any(k in d for k in ["ARDUINO", "CH340", "USB-SERIAL"]) or any(
            v in h for v in ["VID:2341", "VID:2A03", "VID:1A86"]
        ):
            return p.device
    return None


def open_serial(cfg: Config):
    """Ouvre le port série si Arduino détecté. None sinon."""
    port = find_arduino_port()
    if not port:
        print("ℹ️  Aucun Arduino détecté (branche-le si tu veux l’envoi auto).")
        return None
    try:
        ser = serial.Serial(port, cfg.baud, timeout=1)
        time.sleep(2)  # reset auto Uno
        if cfg.verbose_serial:
            print(f"🔌 Arduino connecté sur {port} @ {cfg.baud}")
        return ser
    except Exception as e:
        print("❌ Échec connexion série :", e)
        return None


# =========================
# Vision (RR/GG/BB + retrait fond vert)
# =========================
def red_mask_rrggbb(bgr: np.ndarray, R_MIN: int, K_DOM: float) -> np.ndarray:
    """Masque binaire des pixels 'rouges' via règle RR/GG/BB."""
    B, G, R = cv2.split(bgr)
    Rf = R.astype(np.float32); Gf = G.astype(np.float32); Bf = B.astype(np.float32)
    cond = (R >= R_MIN) & (Rf >= K_DOM * Gf) & (Rf >= K_DOM * Bf)
    return (cond.astype(np.uint8) * 255)


def green_mask_rrggbb(bgr: np.ndarray, G_MIN: int, K_G_DOM: float) -> np.ndarray:
    """Masque binaire des pixels 'verts' (fond) via règle GG/RR/BB."""
    B, G, R = cv2.split(bgr)
    Gf = G.astype(np.float32); Rf = R.astype(np.float32); Bf = B.astype(np.float32)
    cond = (G >= G_MIN) & (Gf >= K_G_DOM * Rf) & (Gf >= K_G_DOM * Bf)
    return (cond.astype(np.uint8) * 255)


def clean_mask(mask: np.ndarray, open_ks: int, close_ks: int) -> np.ndarray:
    """Ouverture/fermeture morphologique pour réduire bruit et combler trous."""
    if open_ks > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ks, open_ks), np.uint8))
    if close_ks > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_ks, close_ks), np.uint8))
    return mask


def largest_component(mask: np.ndarray, min_area: int):
    """Plus grand contour du masque (None si trop petit)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, 0
    c = max(cnts, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    if area < min_area: return None, 0
    return c, area


def ripe_by_area_fraction(mask: np.ndarray, contour, frac_min: float) -> bool:
    """'Mûr' si la fraction de rouge dans la boîte englobante ≥ seuil."""
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y:y+h, x:x+w]
    red_px = int((roi > 0).sum())
    frac = red_px / max(1, (w*h))
    return frac >= frac_min


def max_width_pixels_by_scan(mask: np.ndarray, contour) -> int:
    """
    Largeur maximale en pixels par DOUBLE BOUCLE :
      - pour chaque ligne, on cherche le premier et le dernier pixel '1'
      - largeur ligne = last-first+1 ; on garde la plus grande.
    """
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y:y+h, x:x+w]
    max_span = 0
    for row in range(roi.shape[0]):      # lignes
        first = -1; last = -1
        for col in range(roi.shape[1]):  # colonnes
            if roi[row, col] > 0:
                if first == -1: first = col
                last = col
        if first != -1 and last != -1:
            span = last - first + 1
            if span > max_span: max_span = span
    return max_span


def mm_per_pixel(cfg: Config) -> float:
    """Renvoie mm/pixel (override si >0, sinon REF_MM/REF_PX)."""
    if cfg.override_mm_per_px > 0: return cfg.override_mm_per_px
    return cfg.REF_MM / max(1e-6, cfg.REF_PX)


# =========================
# Envoi Arduino
# =========================
def send_grab_command(ser, width_mm: float, size_label: str):
    """Envoie: GRAB,<SMALL|LARGE>,<width_mm_int>"""
    line = f"GRAB,{size_label},{int(round(width_mm))}\n"
    try:
        ser.write(line.encode("ascii"))
    except Exception as e:
        print("❌ Envoi série échoué :", e)


# =========================
# Boucle principale
# =========================
def main():
    print("✅ Démarrage vision (webcam intégrée, fond vert retiré)…")
    cap = open_camera(CFG)        # ouvre webcam intégrée (0), fallback auto si besoin
    ser = open_serial(CFG)        # peut être None si rien de branché
    mm_per_px = mm_per_pixel(CFG)
    if mm_per_px <= 0:
        print("⚠️ Calibration pixels→mm invalide. On met 1.0 par sécurité.")
        mm_per_px = 1.0

    last_sent = 0.0

    try:
        while True:
            # Reconnexion caméra si besoin
            if cap is None:
                time.sleep(0.2)
                cap = open_camera(CFG)
                continue

            ok, frame = cap.read()
            if not ok:
                print("⚠️ Frame non lue; reconnection caméra…")
                cap.release(); cap = None
                continue

            # Lissage léger
            blur = cv2.GaussianBlur(frame, (5,5), 0)

            # Masques RR/GG/BB
            mask_red = red_mask_rrggbb(blur, CFG.R_MIN, CFG.K_DOM)
            mask_green = green_mask_rrggbb(blur, CFG.G_MIN, CFG.K_G_DOM)

            # Retrait explicite du fond vert
            mask_red_nogreen = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_green))

            # Nettoyage
            mask = clean_mask(mask_red_nogreen, CFG.morph_open_ks, CFG.morph_close_ks)

            # Détection plus grande framboise
            contour, _ = largest_component(mask, CFG.min_area_px)

            ripe = False
            width_px = 0
            width_mm = 0.0
            size_label = "UNKNOWN"

            if contour is not None:
                # maturité = fraction rouge dans la boîte
                ripe = ripe_by_area_fraction(mask, contour, CFG.red_area_frac_min)
                # largeur max (double boucle)
                width_px = max_width_pixels_by_scan(mask, contour)
                width_mm = width_px * mm_per_px
                size_label = "SMALL" if width_mm < CFG.small_large_threshold_mm else "LARGE"

                # Dessins debug
                if CFG.show_window:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0) if ripe else (0,0,255), 2)
                    cv2.putText(frame, f"ripe={int(ripe)} width={width_mm:.1f}mm ({width_px}px) {size_label}",
                                (x, max(0,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                if CFG.show_window:
                    cv2.putText(frame, "No raspberry", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Envoi auto si mûre (avec anti-spam)
            now = time.time()
            if ser is not None and CFG.send_when_ripe and ripe and (now - last_sent >= CFG.min_interval_s):
                send_grab_command(ser, width_mm, size_label)
                last_sent = now

            # Affichage
            if CFG.show_window:
                cv2.imshow("camera", frame)
                cv2.imshow("mask_red", mask)

                # Quitter si l’utilisateur ferme une fenêtre
                if cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1 or \
                   cv2.getWindowProperty("mask_red", cv2.WND_PROP_VISIBLE) < 1:
                    break

            # Clavier
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):  # ESC ou 'q'
                break
            elif k == ord('g'):           # forcer une prise
                if ser is not None:
                    send_grab_command(ser, width_mm, size_label if contour is not None else "UNKNOWN")
            elif k == ord('v'):           # toggle affichage
                CFG.show_window = not CFG.show_window
                if not CFG.show_window:
                    cv2.destroyAllWindows()

            # Si Arduino branché après coup, tente d’ouvrir
            if ser is None:
                ser = open_serial(CFG)

    except KeyboardInterrupt:
        print("\n⛔ Interruption clavier (Ctrl+C).")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()
        print("✨ Fin propre du programme.")


if __name__ == "__main__":
    main()
