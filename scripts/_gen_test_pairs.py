"""生成 LFW 测试 pairs.txt（从已有 LFW 数据中随机采样 600 对）。"""
import os, random
random.seed(42)
lfw_dir = "datasets/lfw"
identities = sorted([d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d))])
identities = [d for d in identities if len(os.listdir(os.path.join(lfw_dir, d))) >= 2]
print(f"identities with >=2 imgs: {len(identities)}")

pairs = []
for _ in range(300):
    name = random.choice(identities)
    imgs = sorted(os.listdir(os.path.join(lfw_dir, name)))
    if len(imgs) < 2:
        continue
    i1, i2 = random.sample(range(1, len(imgs)+1), 2)
    pairs.append(f"{name} {i1} {i2}")
for _ in range(300):
    n1, n2 = random.sample(identities, 2)
    i1 = random.randint(1, len(os.listdir(os.path.join(lfw_dir, n1))))
    i2 = random.randint(1, len(os.listdir(os.path.join(lfw_dir, n2))))
    pairs.append(f"{n1} {i1} {n2} {i2}")

with open("datasets/pairs_test.txt", "w") as f:
    f.write("10 300\n")
    for p in pairs:
        f.write(p + "\n")
print(f"wrote {len(pairs)} pairs to datasets/pairs_test.txt")