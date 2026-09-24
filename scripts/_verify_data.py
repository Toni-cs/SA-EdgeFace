"""验证 pairs.txt 和 CASIA 数据完整性。"""
import os

# pairs.txt
with open("datasets/pairs.txt") as f:
    lines = f.readlines()
print(f"pairs.txt: {len(lines)} lines, first: {lines[0].strip()}")
same = sum(1 for l in lines[1:] if len(l.split()) == 3)
diff = sum(1 for l in lines[1:] if len(l.split()) == 4)
print(f"  same pairs: {same}, diff pairs: {diff}, total: {same+diff}")

# CASIA
casia = "datasets/raw/casia-webface"
dirs = sorted(os.listdir(casia))
print(f"\nCASIA: {len(dirs)} identities")
print(f"  first: {dirs[0]}, last: {dirs[-1]}")
sample = os.path.join(casia, dirs[0])
files = os.listdir(sample)
print(f"  {dirs[0]}: {len(files)} imgs, e.g. {files[:3]}")
total_imgs = sum(len(os.listdir(os.path.join(casia, d))) for d in dirs[:100])
print(f"  first 100 ids: {total_imgs} imgs (avg {total_imgs/100:.1f}/id)")