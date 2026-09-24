import torch, cv2, numpy as np, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qad_face.utils import imread_cn
from qad_face.models.edgeface import DEFAULT_CONFIGS, build_edgeface

model = build_edgeface(DEFAULT_CONFIGS['x05_baseline'])
ckpt = torch.load('weights/ablation_casia/cfg0_final.pt', map_location='cpu')
model.load_state_dict(ckpt['student'], strict=False)
model.eval()
arc_w = ckpt['arcface']['weight'].numpy()
print('arcface weight:', arc_w.shape)

id_dirs = sorted([d for d in os.listdir('datasets/raw/casia-webface') if os.path.isdir(os.path.join('datasets/raw/casia-webface', d))])[:5]
correct = 0; total = 0
w = arc_w / (np.linalg.norm(arc_w, axis=1, keepdims=True) + 1e-8)
for label, name in enumerate(id_dirs):
    imgs = sorted(glob.glob(os.path.join('datasets/raw/casia-webface', name, '*.jpg')))[:3]
    for p in imgs:
        img = imread_cn(p)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112,112)).astype(np.float32)/127.5 - 1.0
        t = torch.from_numpy(np.transpose(img,(2,0,1))).unsqueeze(0)
        with torch.no_grad():
            emb = model.get_embedding(t).numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        pred = int(np.argmax(emb @ w.T))
        correct += (pred == label); total += 1
        print('  id=%s label=%d pred=%d %s' % (name, label, pred, 'OK' if pred==label else 'MISS'))
print('训练集分类Acc: %d/%d' % (correct, total))
