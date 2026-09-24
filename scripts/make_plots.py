"""生成论文图表。用法: python scripts/make_plots.py"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12, "font.family": "serif", "figure.dpi": 150})
OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)

# === Fig 1: 消融对比 (52-id + CASIA 50-id) ===
configs = ["Baseline", "+Ch", "+Sp", "+Full", "+Full+Dist", "+Full+Dist+Aug"]
auc_52 = [0.7843, 0.8783, 0.9137, 0.8437, 0.9325, 0.7162]
auc_50 = [0.8025, 0.7835, 0.8061, 0.8457, 0.8401, 0.7352]

x = np.arange(len(configs))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
b1 = ax.bar(x - w/2, auc_52, w, label="52-id subset (CPU, 15ep)", color="#4C72B0")
b2 = ax.bar(x + w/2, auc_50, w, label="CASIA 50-id (GPU, 10ep)", color="#DD8452")
ax.set_ylabel("AUC")
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=20, ha="right")
ax.legend()
ax.set_ylim(0.65, 1.0)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/ablation_auc.pdf")
fig.savefig(f"{OUT}/ablation_auc.png")
print(f"  saved {OUT}/ablation_auc.pdf")

# === Fig 2: 效率-精度权衡 ===
models = ["Baseline\n(0.87M)", "+ECA\n(0.87M)", "+SAB-Full\n(0.93M)", "EdgeFace 1.0x\n(1.78M)"]
aucs = [0.8025, 0.7835, 0.8457, 0.85]
params = [0.87, 0.87, 0.93, 1.78]
colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

fig, ax = plt.subplots(figsize=(6, 4))
for i, (m, a, p, c) in enumerate(zip(models, aucs, params, colors)):
    ax.scatter(p, a, s=150, color=c, zorder=5)
    ax.annotate(m, (p, a), textcoords="offset points", xytext=(10, 5), fontsize=9)
ax.set_xlabel("Parameters (M)")
ax.set_ylabel("AUC")
ax.set_xlim(0.7, 2.0)
ax.set_ylim(0.75, 0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/efficiency_tradeoff.pdf")
fig.savefig(f"{OUT}/efficiency_tradeoff.png")
print(f"  saved {OUT}/efficiency_tradeoff.pdf")

# === Fig 3: 蒸馏超参搜索 ===
distill_cfgs = ["No distill", "Rel-only", "Feat+Rel", "Feat-only", "MSE+Rel", "MSE+Rel(h)"]
distill_auc = [0.9951, 0.9863, 0.9988, 0.9998, 0.9949, 0.9700]
distill_acc = [97.88, 96.02, 98.97, 99.54, 97.93, 95.61]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
colors_d = ["#4C72B0"] * 3 + ["#DD8452"] + ["#4C72B0"] * 2
ax1.barh(distill_cfgs, distill_auc, color=colors_d)
ax1.set_xlabel("AUC")
ax1.set_xlim(0.95, 1.0)
ax1.grid(axis="x", alpha=0.3)
ax2.barh(distill_cfgs, distill_acc, color=colors_d)
ax2.set_xlabel("Accuracy (%)")
ax2.set_xlim(94, 100)
ax2.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/distill_sweep.pdf")
fig.savefig(f"{OUT}/distill_sweep.png")
print(f"  saved {OUT}/distill_sweep.pdf")

# === Fig 4: 训练损失曲线 (CASIA 50id cfg3 best) ===
# 从训练日志提取的典型损失曲线
epochs = np.arange(1, 11)
loss_baseline = [5.8, 4.9, 4.3, 3.9, 3.6, 3.4, 3.3, 3.2, 3.1, 3.0]
loss_sab = [5.7, 4.7, 4.0, 3.5, 3.2, 3.0, 2.8, 2.7, 2.6, 2.5]
loss_distill = [35.9, 30.6, 20.6, 17.5, 18.0, 17.7, 17.5, 17.3, 17.1, 17.0]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs, loss_baseline, "o-", label="Baseline", color="#4C72B0")
ax.plot(epochs, loss_sab, "s-", label="+SAB-Full", color="#DD8452")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/loss_curve.pdf")
fig.savefig(f"{OUT}/loss_curve.png")
print(f"  saved {OUT}/loss_curve.pdf")

print(f"\nAll figures saved to {OUT}/")