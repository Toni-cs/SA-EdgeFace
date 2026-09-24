import os
import py7zr

src = "datasets/raw/casia-webface.7z"
print("=== 查看 CASIA 7z 结构 ===", flush=True)
with py7zr.SevenZipFile(src, mode="r") as z:
    names = z.getnames()
    print(f"total entries: {len(names)}", flush=True)
    for n in names[:20]:
        print(f"  {n}", flush=True)
    print("...", flush=True)
    print("=== 解压中（可能需几分钟）===", flush=True)
    z.extractall(path="datasets/raw/")
print("[DONE] 解压完成", flush=True)
for d in os.listdir("datasets/raw"):
    p = os.path.join("datasets/raw", d)
    if os.path.isdir(p):
        sub = os.listdir(p)[:5]
        print(f"  {d}/ -> {sub}", flush=True)
    else:
        print(f"  {d} ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)