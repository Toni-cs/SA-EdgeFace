"""
生成训练曲线图（loss + lr vs epoch），6 配置叠加。
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES_DIR = "results/training_curves"
OUTPUT_PATH = "paper/figs/training_curves.pdf"

CONFIGS = ["cfg0", "cfg1", "cfg2", "cfg3", "cfg4", "cfg5"]
LABELS = {
    "cfg0": "C0 baseline",
    "cfg1": "C1 +channel",
    "cfg2": "C2 +spatial",
    "cfg3": "C3 +full SAB",
    "cfg4": "C4 +distill",
    "cfg5": "C5 +aug",
}
COLORS = {
    "cfg0": "#1f77b4",
    "cfg1": "#ff7f0e",
    "cfg2": "#2ca02c",
    "cfg3": "#d62728",
    "cfg4": "#9467bd",
    "cfg5": "#8c564b",
}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for cfg_id in CONFIGS:
        path = os.path.join(CURVES_DIR, f"{cfg_id}_curve.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        epochs = data["epochs"]
        ep = [e["epoch"] for e in epochs]
        loss = [e["avg_loss"] for e in epochs]
        lr = [e["lr"] for e in epochs]
        label = LABELS.get(cfg_id, cfg_id)
        color = COLORS.get(cfg_id, "gray")
        ax1.plot(ep, loss, label=label, color=color, linewidth=1.5)
        ax2.plot(ep, lr, label=label, color=color, linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 30)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 30)
    ax2.set_yscale("log")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()