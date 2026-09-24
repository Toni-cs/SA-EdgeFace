import os, glob
print("=== CASIA 50 checkpoints ===")
for f in sorted(glob.glob("weights/casia50/cfg*_final.pt")):
    print(f"  {os.path.basename(f)}: {os.path.getsize(f)/1024:.0f}KB")
p = "weights/casia50/results.txt"
if os.path.exists(p):
    print("\n=== casia50 results.txt ===")
    print(open(p).read())

print("=== face_dataset_52 ablation checkpoints ===")
for f in sorted(glob.glob("weights/ablation_final/cfg*_final.pt")):
    print(f"  {os.path.basename(f)}: {os.path.getsize(f)/1024:.0f}KB")

print("\n=== results files ===")
for f in sorted(glob.glob("results/*.txt")):
    print(f"  {f}")