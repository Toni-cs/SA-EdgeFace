"""生成 50 对 mini pairs 用于快速验证 LFW pipeline。"""
import os, random
random.seed(42)
lfw_dir = "datasets/lfw"
identities = sorted([d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d))])
identities = [d for d in identities if len(os.listdir(os.path.join(lfw_dir, d))) >= 2]

pairs = []
for _ in range(25):
    name = random.choice(identities)
    imgs = sorted(os.listdir(os.path.join(lfw_dir, name)))
    i1, i2 = random.sample(range(1, len(imgs)+1), 2)
    pairs.append(f"{name} {i1} {i2}")
for _ in range(25):
    n1, n2 = random.sample(identities, 2)
    i1 = random.randint(1, len(os.listdir(os.path.join(lfw_dir, n1))))
    i2 = random.randint(1, len(os.listdir(os.path.join(lfw_dir, n2))))
    pairs.append(f"{n1} {i1} {n2} {i2}")

with open("datasets/pairs_mini.txt", "w") as f:
    f.write("10 300\n")
    for p in pairs:
        f.write(p + "\n")
print(f"wrote {len(pairs)} pairs to datasets/pairs_mini.txt")