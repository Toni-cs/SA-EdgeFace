"""Independent audit re-check: recompute LFW stats + tail behaviour for cfg0-cfg5."""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface
from qad_face.evaluation.lfw_eval import load_pairs, get_image_path
from sklearn.metrics import roc_curve, auc

LFW_DIR = "datasets/lfw_aligned"
PAIRS_PATH = "datasets/pairs.txt"
SAVE_DIR = "weights/ablation_casia"

BATCH = [
    ("cfg0", "x05_baseline", "baseline"),
    ("cfg1", "x05_sab_channel", "+channel"),
    ("cfg2", "x05_sab_spatial", "+spatial"),
    ("cfg3", "x05_sab_full", "+fullSAB"),
    ("cfg4", "x05_sab_full", "+distill"),
    ("cfg5", "x05_sab_full", "+aug"),
]


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pairs = load_pairs(PAIRS_PATH)
    needed = sorted({get_image_path(LFW_DIR, n1, i1) for n1, i1, n2, i2, s in pairs} |
                    {get_image_path(LFW_DIR, n2, i2) for n1, i1, n2, i2, s in pairs})
    cache = {}
    for p in needed:
        if os.path.exists(p):
            img = imread_cn(p)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112)).astype(np.float32) / 127.5 - 1.0
                cache[p] = np.transpose(img, (2, 0, 1))
    print(f"device={device} images={len(cache)}/{len(needed)} pairs={len(pairs)}", flush=True)

    labels = np.array([1 if s else 0 for _, _, _, _, s in pairs])
    out = {}
    for save_name, cfg_name, desc in BATCH:
        ck = os.path.join(SAVE_DIR, f"{save_name}_final.pt")
        model = build_edgeface(DEFAULT_CONFIGS[cfg_name]).to(device)
        model.load_state_dict(torch.load(ck, map_location=device)["student"], strict=False)
        model.eval()
        paths = [p for p in needed if p in cache]
        feats = np.zeros((len(paths), 512), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(paths), 256):
                b = np.array([cache[p] for p in paths[s:s + 256]], dtype=np.float32)
                feats[s:s + len(b)] = model.get_embedding(torch.from_numpy(b).to(device)).cpu().numpy()[:, :512]
        fc = dict(zip(paths, feats))
        f1 = np.array([fc.get(get_image_path(LFW_DIR, n1, i1), np.zeros(512, np.float32)) for n1, i1, n2, i2, s in pairs])
        f2 = np.array([fc.get(get_image_path(LFW_DIR, n2, i2), np.zeros(512, np.float32)) for n1, i1, n2, i2, s in pairs])
        sims = np.sum(f1 * f2, axis=1) / (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-8)

        fpr, tpr, thr = roc_curve(labels, sims)
        neg = np.sort(sims[labels == 0])[::-1]
        pos = np.sort(sims[labels == 1])[::-1]

        def tar_at(far):
            i = np.searchsorted(fpr, far)
            return float(tpr[i]) if i < len(tpr) else 0.0

        # 10-fold CV (exact copy of repo protocol)
        n_same, n_diff = 3000, 3000
        same_idx, diff_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]
        accs = []
        for k in range(10):
            ts = same_idx[k * 300:(k + 1) * 300]
            td = diff_idx[k * 300:(k + 1) * 300]
            te = np.concatenate([ts, td])
            va = np.setdiff1d(np.arange(len(pairs)), te)
            vf, vt, vth = roc_curve(labels[va], sims[va])
            best, bt = 0, 0.0
            for t in vth:
                a = np.mean((sims[va] >= t).astype(int) == labels[va])
                if a > best:
                    best, bt = a, t
            accs.append(np.mean((sims[te] >= bt).astype(int) == labels[te]))
        accs = np.array(accs)

        # honest zero-false-positive operation point
        thr0 = neg[0] + 1e-12
        tar0 = float(np.mean(sims[labels == 1] >= thr0))

        out[desc] = dict(
            cv_acc=float(accs.mean()), cv_std=float(accs.std()),
            oracle_acc=float(max(np.mean((sims >= t).astype(int) == labels) for t in thr)),
            auc=float(auc(fpr, tpr)),
            n_correct_oracle=int(round(float(max(np.mean((sims >= t).astype(int) == labels) for t in thr)) * 6000)),
            tar1e3_searchsorted=tar_at(1e-3), tar1e4_searchsorted=tar_at(1e-4),
            top_neg=neg[:6].tolist(),
            far_at_reported_1e4_threshold=float(np.mean(neg > thr[min(np.searchsorted(fpr, 1e-4), len(thr) - 1)])),
            thr_at_1e4=float(thr[min(np.searchsorted(fpr, 1e-4), len(thr) - 1)]),
            tar_zero_fp=tar0,
            tar1e3_exact=float(tpr[np.where(fpr <= 1e-3)[0][-1]]),
        )
        print(desc, json.dumps(out[desc], indent=1), flush=True)
        del model
        torch.cuda.empty_cache()

    with open("results/_audit_recheck.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/_audit_recheck.json")


if __name__ == "__main__":
    main()
