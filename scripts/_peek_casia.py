import py7zr

with py7zr.SevenZipFile("datasets/raw/casia-webface.7z", mode="r") as z:
    names = z.getnames()
    print(f"total: {len(names)}", flush=True)
    files = [n for n in names if not n.endswith("/")]
    print(f"files: {len(files)}", flush=True)
    print("first 15:", flush=True)
    for n in files[:15]:
        print(f"  {n}", flush=True)
    print("last 5:", flush=True)
    for n in files[-5:]:
        print(f"  {n}", flush=True)
    bins = [n for n in files if n.endswith(".bin") or n.endswith(".rec") or n.endswith(".idx")]
    print(f"\nbin/rec files: {bins[:10]}", flush=True)
    import os
    exts = {}
    for n in files:
        e = os.path.splitext(n)[1].lower()
        exts[e] = exts.get(e, 0) + 1
    print(f"ext counts: {sorted(exts.items(), key=lambda x: -x[1])[:10]}", flush=True)