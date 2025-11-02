#!/usr/bin/env python3
"""
Demo script to showcase all the beautiful animations from train_with_negatives.py
This runs without needing ultralytics installed.
"""

import sys
import time
import random
import itertools
import threading

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

    time.sleep(0.5)


def show_system_check():
    """Animated system check sequence."""
    checks = [
        ("Initializing Neural Networks", 0.6, True),
        ("Loading YOLO Architecture", 0.5, True),
        ("Checking GPU Availability", 0.7, True),
        ("Validating Dataset Structure", 0.8, True),
        ("Preparing Augmentation Pipeline", 0.6, True),
        ("Optimizing Memory Allocation", 0.5, True),
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


def show_dataset_statistics():
    """Show animated dataset statistics."""
    print(f"\n\033[1m📊 DATASET STATISTICS 📊\033[0m")
    print("─" * 80)

    # Simulate counting animation
    def count_animation(target, label, emoji):
        """Animate counting up to target value."""
        steps = 30
        for i in range(steps + 1):
            current = int(target * i / steps)
            sys.stdout.write(f"\r  {emoji} {label}: {current:,}")
            sys.stdout.flush()
            time.sleep(0.03)
        print()

    count_animation(1247, "Total training images", "📸")
    count_animation(1089, "Images with labels (positives)", "✅")
    count_animation(158, "Images without labels (negatives)", "❌")

    # Visual ratio bar
    neg_percentage = 12.7
    print(f"\n  📈 Negative ratio: {create_gradient_bar(neg_percentage, width=40)}")
    print("─" * 80)


def show_configuration():
    """Show training configuration with visual bars."""
    print(f"\n\033[1m⚙️  TRAINING CONFIGURATION\033[0m")
    print("╔" + "═" * 78 + "╗")
    print(f"║  {'Parameter':<30} {'Value':<45} ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  {'Model':<30} \033[96m{'yolov8s.pt':<45}\033[0m ║")
    print(f"║  {'Epochs':<30} \033[93m{'120':<45}\033[0m ║")
    print(f"║  {'Batch size':<30} \033[93m{'8':<45}\033[0m ║")
    print(f"║  {'Device':<30} \033[92m{'MPS (Apple Silicon)':<45}\033[0m ║")
    print(f"║  {'Optimizer':<30} \033[92m{'AdamW':<45}\033[0m ║")
    print("╚" + "═" * 78 + "╝")

    print(f"\n\033[1m🎨 AUGMENTATION PIPELINE\033[0m")
    print("╔" + "═" * 78 + "╗")

    # Visual representation of augmentation strength
    aug_params = [
        ("HSV Saturation", 60),
        ("HSV Value", 45),
        ("Rotation", 47),
        ("Translation", 20),
        ("Scale", 35),
        ("Mosaic", 60),
        ("Mixup", 15),
        ("Copy-Paste", 40),
        ("Random Erasing", 25),
    ]

    for param_name, percentage in aug_params:
        bar = create_gradient_bar(percentage, width=30, colors=True)
        print(f"║  {param_name:<20} {bar:<52} ║")
        time.sleep(0.1)  # Animate each bar appearing

    print("╚" + "═" * 78 + "╝")


def simulate_training():
    """Simulate training progress."""
    print("\n\033[1m⚡ TRAINING IN PROGRESS ⚡\033[0m")
    print("═" * 80)

    epochs = 10  # Shortened for demo
    for epoch in range(1, epochs + 1):
        # Progress bar for epoch
        percentage = (epoch / epochs) * 100
        bar = create_gradient_bar(percentage, width=50)

        # Random metrics that improve over time
        loss = 1.5 - (epoch * 0.12) + random.uniform(-0.05, 0.05)
        map50 = 0.3 + (epoch * 0.06) + random.uniform(-0.02, 0.02)

        print(f"\n🔄 Epoch {epoch}/{epochs}")
        print(f"   Progress: {bar}")
        print(f"   Loss: \033[93m{loss:.4f}\033[0m  |  mAP50: \033[92m{map50:.4f}\033[0m")

        # Simulate epoch duration with spinner
        spinner = AnimatedSpinner(f"Training batches", style="pulse")
        spinner.start()
        time.sleep(0.8)
        spinner.stop()


def show_final_metrics():
    """Show final metrics with animations."""
    print("\n\n" + "═" * 80)

    # Blinking completion banner
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

    metric_list = [
        ("mAP50", 0.847),
        ("mAP50-95", 0.623),
        ("Precision", 0.891),
        ("Recall", 0.782),
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

        # Animate the value counting up
        bar_width = 25
        steps = 20
        for i in range(steps + 1):
            current = value * i / steps
            filled = int(bar_width * current)
            display_bar = "█" * filled + "░" * (bar_width - filled)
            sys.stdout.write(f"\r║  {emoji} {metric_name:<22} {color}{current:>6.4f}\033[0m             {color}{display_bar}\033[0m      ║")
            sys.stdout.flush()
            time.sleep(0.04)
        print()

    print("╚" + "═" * 78 + "╝")

    # Final celebration
    print("\n")
    celebration = [
        "🎉", "🎊", "✨", "🌟", "💫", "⭐", "🎯", "🚀", "🏆", "👏"
    ]
    for i in range(3):
        line = " ".join(random.choice(celebration) for _ in range(20))
        print(f"\033[{random.randint(91,96)}m{line}\033[0m")
        time.sleep(0.15)

    print("\n" + "═" * 80)
    animated_typewriter("Training session completed successfully! 🎉", delay=0.02, color_code="\033[1;92m")
    print("═" * 80 + "\n")


def main():
    """Run the complete animation demo."""
    # 1. Splash screen
    show_splash_screen()

    # 2. System check
    show_system_check()

    # 3. Dataset statistics
    show_dataset_statistics()

    # 4. Configuration
    show_configuration()

    # 5. Countdown
    print("\n🚀 Starting training in...")
    for i in range(3, 0, -1):
        sys.stdout.write(f"\r   {i}... ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r   GO! 🎯\n\n")
    sys.stdout.flush()

    # 6. Simulate training
    simulate_training()

    # 7. Final metrics
    show_final_metrics()


if __name__ == "__main__":
    main()
