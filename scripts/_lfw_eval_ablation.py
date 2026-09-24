"""LFW 标准评测：用 pairs.txt 对 6 个消融模型评测。"""
import os, sys, time, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path, evaluate_lfw

LFW = "datasets/lfw"
PAIRS = "datasets/pairs.txt"
N_PAIRS = 600  # 前 600 对，沙箱限制

def load_model(cfg, ckpt_path):
    model = build_edgeface(DEFAULT_CONFIGS[cfg])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["student"], strict=False)
    model.eval()
    return model

def extract(model, path):
    if not os.path.exists(path):
        return np.zeros(512, dtype=np.float32)
    img = Image.open(path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0)
    with torch.no_grad():
        return model.get_embedding(t).numpy().flatten()

pairs_all = load_pairs(PAIRS)
pairs = pairs_all[:N_PAIRS]
print(f"LFW pairs: {len(pairs)} (of {len(pairs_all)})", flush=True)

needed = set()
for n1, i1, n2, i2, s in pairs:
    needed.add(get_image_path(LFW, n1, i1))
    needed.add(get_image_path(LFW, n2, i2))
print(f"unique images needed: {len(needed)}", flush=True)

CONFIGS = [
    ("x05_baseline",    "cfg0", "baseline"),
    ("x05_sab_channel", "cfg1", "+C1 channel"),
    ("x05_sab_spatial", "cfg2", "+C1 spatial"),
    ("x05_sab_full",    "cfg3", "+C1 full SAB"),
    ("x05_sab_full",    "cfg4", "+C1+C2 distill"),
    ("x05_sab_full",    "cfg5", "+C1+C2+C3 aug"),
]

results = []
t0 = time.time()
for cfg, save, desc in CONFIGS:
    ckpt = f"weights/ablation_final/{save}_final.pt"
    if not os.path.exists(ckpt):
        print(f"  SKIP {desc}: {ckpt} not found")
        continue
    model = load_model(cfg, ckpt)
    cache = {}
    for p in sorted(needed):
        cache[p] = extract(model, p)
    feats, same_list = [], []
    for n1, i1, n2, i2, s in pairs:
        feats.append(cache[get_image_path(LFW, n1, i1)])
        feats.append(cache[get_image_path(LFW, n2, i2)])
        same_list.append(s)
    r = evaluate_lfw(np.array(feats), pairs, same_list)
    print(f"  {desc:<20} Acc={r['accuracy']*100:.2f}% AUC={r['auc']:.4f}", flush=True)
    results.append((desc, r['accuracy'], r['auc']))

print(f"\n{'='*50}")
print(f"  LFW 评测 ({N_PAIRS} pairs, 50-id trained)")
print(f"{'='*50}")
for desc, acc, auc in results:
    print(f"  {desc:<20} Acc={acc*100:.2f}% AUC={auc:.4f}")
print(f"{'='*50}")
print(f"[DONE] {time.time()-t0:.1f}s", flush=True)