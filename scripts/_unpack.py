import os
import zipfile

print("=== 解压 buffalo_s ===", flush=True)
with zipfile.ZipFile("weights/buffalo_s.zip") as z:
    z.extractall("weights/models/buffalo_s")
onnx = [f for f in os.listdir("weights/models/buffalo_s") if f.endswith(".onnx")]
print(f"buffalo_s onnx: {onnx}", flush=True)

print("=== 解压 LFW ===", flush=True)
with zipfile.ZipFile("datasets/LFW人脸数据集.zip") as z:
    z.extractall("datasets/_lfw_tmp")

inner = "datasets/_lfw_tmp/lfw-deepfunneled/lfw-deepfunneled"
if os.path.isdir(inner):
    if os.path.isdir("datasets/lfw"):
        import shutil
        shutil.rmtree("datasets/lfw")
    os.rename(inner, "datasets/lfw")
    import shutil
    shutil.rmtree("datasets/_lfw_tmp", ignore_errors=True)
n_ids = len(os.listdir("datasets/lfw"))
print(f"lfw: {n_ids} identities -> datasets/lfw", flush=True)
print("[DONE]", flush=True)