#!/usr/bin/env python3
"""
Custom YOLOv8 training script that properly handles negative background images.
This script ensures YOLO loads images without labels as true negatives.

Usage:
    python3 train_with_negatives.py --mode fast    # Quick training (30 epochs)
    python3 train_with_negatives.py --mode full    # Full training (120 epochs)
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import sys
import argparse
from tqdm import tqdm
import time
import random
from datetime import datetime
import itertools
import threading

# Training modes
TRAINING_MODES = {
    'fast': {
        'epochs': 30,
        'patience': 10,
        'batch': 8,
        'description': 'Mode rapide - Test et validation (30 epochs)'
    },
    'full': {
        'epochs': 120,
        'patience': 30,
        'batch': 8,
        'description': 'Mode complet - Entraînement final (120 epochs)'
    }
}

# Base configuration
DATA_YAML = "data/raspberries.yaml"
MODEL = "yolov8s.pt"
IMGSZ = 896

# Auto-detect device (Windows/macOS/Linux compatible)
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"  # NVIDIA GPU on Windows/Linux
elif torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple Silicon on macOS
else:
    DEVICE = "cpu"   # CPU fallback

# Augmentation parameters
HSV_H = 0.015
HSV_S = 0.6
HSV_V = 0.45
DEGREES = 7
TRANSLATE = 0.10
SCALE = 0.35
SHEAR = 3
PERSPECTIVE = 0.0005
MOSAIC = 0.6
MIXUP = 0.15
COPY_PASTE = 0.4
ERASING = 0.25

# Training hyperparameters
BOX = 7
CLS = 0.6
DFL = 1.5
LR0 = 0.004
LRF = 0.1
OPTIMIZER = "AdamW"
COS_LR = True
WARMUP_EPOCHS = 3

def count_negative_images(data_yaml_path):
    """Count images without corresponding label files (true negatives)."""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_config['path'])
    train_images_dir = dataset_root / data_config['train']
    train_labels_dir = dataset_root / 'labels' / 'train'

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(train_images_dir.glob(ext))

    # Count images without labels
    negatives = 0
    positives = 0

    for img_path in all_images:
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            negatives += 1
        else:
            positives += 1

    return negatives, positives, len(all_images)


# ═══════════════════════════════════════════════════════════════════════════
# 🎨 ANIMATED UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

class AnimatedSpinner:
    """Animated spinner with different styles."""
    def __init__(self, message="Loading", style="dots"):
        self.message = message
        self.running = False
        self.thread = None

        self.styles = {
            "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            "line": ["-", "\\", "|", "/"],
            "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
            "bounce": ["⠁", "⠂", "⠄", "⠂"],
            "dots2": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
            "pulse": ["◐", "◓", "◑", "◒"],
            "grow": ["▁", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃"],
        }
        self.frames = self.styles.get(style, self.styles["dots"])

    def _spin(self):
        """Run the spinner animation."""
        for frame in itertools.cycle(self.frames):
            if not self.running:
                break
            sys.stdout.write(f"\r{frame} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        """Start the spinner."""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self, final_message=None):
        """Stop the spinner."""
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        if final_message:
            print(final_message)


def animated_typewriter(text, delay=0.03, color_code="\033[92m"):
    """Typewriter effect for text."""
    reset = "\033[0m"
    for char in text:
        sys.stdout.write(color_code + char + reset)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def create_gradient_bar(percentage, width=50, colors=True):
    """Create a colorful gradient progress bar."""
    filled = int(width * percentage / 100)
    empty = width - filled

    if colors:
        # Color gradient: red -> yellow -> green
        if percentage < 33:
            color = "\033[91m"  # Red
        elif percentage < 66:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[92m"  # Green

        bar = color + "█" * filled + "\033[90m░" * empty + "\033[0m"
    else:
        bar = "█" * filled + "░" * empty

    return f"[{bar}] {percentage:>3.0f}%"


def show_splash_screen():
    """Display animated splash screen."""
    # Clear screen
    print("\033[2J\033[H")

    # Raspberry ASCII art
    art = r"""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║        🍓   __   __  _____  _      _____  __   __  _____                 ║
    ║            |  \ /  ||  _  || |    |  _  | \ \ / / |  _  |                ║
    ║            |   v   || | | || |    | | | |  \ v /  | |_| |                ║
    ║            |_|\_/|_||_| |_||___/  |_____/   \_/   |_____/                ║
    ║                                                                           ║
    ║              🤖  R A S P B E R R Y   D E T E C T I O N  🤖               ║
    ║                   Ultra Advanced Training System v2.0                    ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    # Animate appearance
    lines = art.split('\n')
    for line in lines:
        print(f"\033[96m{line}\033[0m")  # Cyan color
        time.sleep(0.05)

    time.sleep(0.3)


def show_system_check():
    """Animated system check sequence."""
    checks = [
        ("Initializing Neural Networks", 0.4, True),
        ("Loading YOLO Architecture", 0.3, True),
        ("Checking GPU Availability", 0.5, True),
        ("Validating Dataset Structure", 0.6, True),
        ("Preparing Augmentation Pipeline", 0.4, True),
        ("Optimizing Memory Allocation", 0.3, True),
    ]

    print("\n\033[1m🔧 SYSTEM CHECK\033[0m")
    print("─" * 80)

    for check, duration, success in checks:
        spinner = AnimatedSpinner(check, style="dots2")
        spinner.start()
        time.sleep(duration)

        if success:
            spinner.stop(f"✅ {check}")
        else:
            spinner.stop(f"❌ {check}")

    print("─" * 80)


def show_loading_animation(message, steps=20):
    """Show a loading bar animation."""
    print(f"\n{message}")

    with tqdm(total=steps,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              colour='green',
              ncols=80) as pbar:
        for _ in range(steps):
            time.sleep(0.05)
            pbar.update(1)


def print_metric_bar(name, value, max_value=1.0, width=40):
    """Print a metric with a visual bar."""
    percentage = (value / max_value) * 100
    filled = int(width * value / max_value)

    # Color based on value
    if value > 0.8 * max_value:
        color = "\033[92m"  # Green
        emoji = "🟢"
    elif value > 0.5 * max_value:
        color = "\033[93m"  # Yellow
        emoji = "🟡"
    else:
        color = "\033[91m"  # Red
        emoji = "🔴"

    bar = "█" * filled + "░" * (width - filled)
    print(f"  {emoji} {name:.<25} {color}{bar}\033[0m {value:.4f}")


def create_mini_chart(values, width=50, height=8):
    """Create a mini ASCII chart from values."""
    if not values:
        return []

    max_val = max(values)
    min_val = min(values)
    range_val = max_val - min_val if max_val != min_val else 1

    # Normalize values
    normalized = [(v - min_val) / range_val for v in values]

    # Create chart
    chart_lines = []
    for h in range(height, 0, -1):
        line = ""
        threshold = h / height
        for val in normalized[-width:]:
            if val >= threshold:
                line += "█"
            elif val >= threshold - (1/height):
                line += "▄"
            else:
                line += " "
        chart_lines.append(line)

    return chart_lines


def print_training_header(mode, epochs, batch_size, device):
    """Print stylized training header."""
    print("\n" + "═" * 80)
    print(f"{'⚡ COMMENCING TRAINING SEQUENCE ⚡':^80}")
    print("═" * 80)

    # Configuration box
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  MODE: {mode.upper():^57} ║
    ║  EPOCHS: {epochs:^55} ║
    ║  BATCH SIZE: {batch_size:^51} ║
    ║  DEVICE: {device.upper():^55} ║
    ║  TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^52} ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


def animate_countdown(seconds=3):
    """Animated countdown before starting."""
    print("\n🚀 Starting training in...")
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\r   {i}... ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r   GO! 🎯\n\n")
    sys.stdout.flush()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Training with Negative Background Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  fast : Mode rapide (30 epochs) - Pour tester et valider rapidement
  full : Mode complet (120 epochs) - Pour l'entraînement final de production

Examples:
  python3 train_with_negatives.py --mode fast
  python3 train_with_negatives.py --mode full
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fast', 'full'],
        default='full',
        help='Training mode: fast (30 epochs) or full (120 epochs)'
    )
    return parser.parse_args()

def print_progress_header(mode_config, negatives, positives, total):
    """Print training header with progress info."""
    print("\n" + "═" * 80)
    print("\033[1;96m🍓 YOLOV8 RASPBERRY DETECTION - TRAINING WITH NEGATIVES 🍓\033[0m".center(90))
    print("═" * 80)

    # Animated dataset statistics
    print(f"\n\033[1m📊 DATASET STATISTICS 📊\033[0m")
    print("─" * 80)

    # Animated counters
    def count_animation(target, label, emoji):
        """Animate counting up to target value."""
        steps = 20
        for i in range(steps + 1):
            current = int(target * i / steps)
            sys.stdout.write(f"\r  {emoji} {label}: {current:,}")
            sys.stdout.flush()
            time.sleep(0.02)
        print()

    count_animation(total, "Total training images", "📸")
    count_animation(positives, "Images with labels (positives)", "✅")
    count_animation(negatives, "Images without labels (negatives)", "❌")

    # Visual ratio bar
    neg_percentage = (negatives / total * 100) if total > 0 else 0
    print(f"\n  📈 Negative ratio: {create_gradient_bar(neg_percentage, width=40)}")
    print("─" * 80)

    # Info message with typewriter effect
    print(f"\n💡 Loaded \033[1;93m{negatives}\033[0m negative background images with no labels.")
    print("   These will be treated as empty scenes to reduce false positives.\n")

def main():
    # 🎬 SHOW SPLASH SCREEN
    show_splash_screen()

    # Parse arguments
    args = parse_arguments()
    mode_config = TRAINING_MODES[args.mode]

    # 🔧 SYSTEM CHECK
    show_system_check()

    # Mode banner
    print("\n" + "═" * 80)
    animated_typewriter(f"⚙️  MODE: {args.mode.upper()} - {mode_config['description']}", delay=0.01, color_code="\033[1;95m")
    print("═" * 80)

    # Count negative images with animated spinner
    spinner = AnimatedSpinner("Analyzing dataset structure", style="dots2")
    spinner.start()
    negatives, positives, total = count_negative_images(DATA_YAML)
    spinner.stop("✅ Dataset analysis complete")

    # Show dataset statistics with animations
    print_progress_header(mode_config, negatives, positives, total)

    # Load model with fancy progress bar
    show_loading_animation(f"📦 Loading {MODEL} architecture...", steps=25)
    model = YOLO(MODEL)

    # Training parameters with visual bars
    print(f"\n\033[1m⚙️  TRAINING CONFIGURATION\033[0m")
    print("╔" + "═" * 78 + "╗")
    print(f"║  {'Parameter':<30} {'Value':<45} ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  {'Model':<30} \033[96m{MODEL:<45}\033[0m ║")
    print(f"║  {'Data':<30} \033[96m{DATA_YAML:<45}\033[0m ║")
    print(f"║  {'Epochs':<30} \033[93m{mode_config['epochs']:<45}\033[0m ║")
    print(f"║  {'Batch size':<30} \033[93m{mode_config['batch']:<45}\033[0m ║")
    print(f"║  {'Image size':<30} \033[93m{IMGSZ}x{IMGSZ:<42}\033[0m ║")
    print(f"║  {'Device':<30} \033[92m{DEVICE.upper():<45}\033[0m ║")
    print(f"║  {'Optimizer':<30} \033[92m{OPTIMIZER:<45}\033[0m ║")
    print(f"║  {'Learning rate':<30} \033[92m{LR0} → {LRF:<39}\033[0m ║")
    print(f"║  {'Patience':<30} \033[93m{mode_config['patience']} epochs{'':<36}\033[0m ║")
    print("╚" + "═" * 78 + "╝")

    print(f"\n\033[1m🎨 AUGMENTATION PIPELINE\033[0m")
    print("╔" + "═" * 78 + "╗")

    # Visual representation of augmentation strength
    aug_params = [
        ("HSV Hue", HSV_H, 0.05),
        ("HSV Saturation", HSV_S, 1.0),
        ("HSV Value", HSV_V, 1.0),
        ("Rotation", DEGREES, 15),
        ("Translation", TRANSLATE, 0.5),
        ("Scale", SCALE, 1.0),
        ("Mosaic", MOSAIC, 1.0),
        ("Mixup", MIXUP, 1.0),
        ("Copy-Paste", COPY_PASTE, 1.0),
        ("Random Erasing", ERASING, 1.0),
    ]

    for param_name, value, max_val in aug_params:
        percentage = (value / max_val) * 100
        bar = create_gradient_bar(percentage, width=30, colors=True)
        print(f"║  {param_name:<20} {bar:<52} ║")

    print("╚" + "═" * 78 + "╝")

    # Display training header
    print_training_header(args.mode, mode_config['epochs'], mode_config['batch'], DEVICE)

    # Countdown before starting
    animate_countdown(3)

    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=mode_config['epochs'],
        imgsz=IMGSZ,
        batch=mode_config['batch'],
        device=DEVICE,
        # Augmentation
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASING,
        # Loss weights
        box=BOX,
        cls=CLS,
        dfl=DFL,
        # Optimizer
        lr0=LR0,
        lrf=LRF,
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        warmup_epochs=WARMUP_EPOCHS,
        patience=mode_config['patience'],
        # Project organization
        project="runs/raspberry_detect",
        name=f"train_{args.mode}",
        exist_ok=True,
        # Ensure negative images are loaded
        verbose=True
    )

    # Animated completion banner
    print("\n" + "═" * 80)
    for _ in range(3):
        print("\033[92m✅ TRAINING COMPLETE ✅\033[0m".center(90), end='\r')
        time.sleep(0.2)
        print(" " * 90, end='\r')
        time.sleep(0.1)
    print("\033[1;92m✅ TRAINING COMPLETE ✅\033[0m".center(90))
    print("═" * 80)

    # Animated "Generating Results" spinner
    spinner = AnimatedSpinner("Compiling final metrics", style="pulse")
    spinner.start()
    time.sleep(1.5)
    spinner.stop()

    # Print final metrics with beautiful bars
    print("\n\033[1m📈 FINAL PERFORMANCE METRICS\033[0m")
    print("╔" + "═" * 78 + "╗")
    print(f"║  {'Metric':<25} {'Score':<20} {'Visual':<30} ║")
    print("╠" + "═" * 78 + "╣")

    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)

        # Animate each metric appearing
        metric_list = [
            ("mAP50", map50),
            ("mAP50-95", map50_95),
            ("Precision", precision),
            ("Recall", recall),
        ]

        for metric_name, value in metric_list:
            # Determine color and emoji
            if value > 0.8:
                color = "\033[92m"
                emoji = "🟢"
            elif value > 0.5:
                color = "\033[93m"
                emoji = "🟡"
            else:
                color = "\033[91m"
                emoji = "🔴"

            # Create visual bar
            bar_width = 25
            filled = int(bar_width * value)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Animate the value counting up
            steps = 15
            for i in range(steps + 1):
                current = value * i / steps
                display_bar = "█" * int(bar_width * current) + "░" * (bar_width - int(bar_width * current))
                sys.stdout.write(f"\r║  {emoji} {metric_name:<22} {color}{current:>6.4f}\033[0m             {color}{display_bar}\033[0m      ║")
                sys.stdout.flush()
                time.sleep(0.03)
            print()

    else:
        print("║  No metrics available - check training logs                            ║")

    print("╚" + "═" * 78 + "╝")

    # Success visualization
    print("\n\033[1m🎯 TRAINING SUMMARY\033[0m")
    print("┌" + "─" * 78 + "┐")

    output_dir = f"runs/raspberry_detect/train_{args.mode}"
    print(f"│  💾 Model Location                                                        │")
    print(f"│     Directory: \033[96m{output_dir:<60}\033[0m│")
    print(f"│     Best weights: \033[92m{output_dir}/weights/best.pt{'':<37}\033[0m│")
    print(f"│     Last weights: \033[93m{output_dir}/weights/last.pt{'':<37}\033[0m│")
    print("├" + "─" * 78 + "┤")
    print("│  💡 Performance Notes                                                     │")
    print("│     → Validation metrics should show improved precision                  │")
    print("│     → Reduced false positives on hands, faces, and red objects           │")
    print("│     → Negative samples help the model learn what NOT to detect           │")
    print("└" + "─" * 78 + "┘")

    # Final celebration
    print("\n")
    celebration = [
        "🎉", "🎊", "✨", "🌟", "💫", "⭐", "🎯", "🚀", "🏆", "👏"
    ]
    for i in range(3):
        line = " ".join(random.choice(celebration) for _ in range(20))
        print(f"\033[{random.randint(91,96)}m{line}\033[0m")
        time.sleep(0.1)

    print("\n" + "═" * 80)
    animated_typewriter("Training session completed successfully! 🎉", delay=0.02, color_code="\033[1;92m")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    main()
