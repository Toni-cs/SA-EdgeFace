import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")
    try:
        x = torch.randn(2, 2).cuda()
        y = x @ x
        print(f"GPU matmul OK: {y.shape}")
    except Exception as e:
        print(f"GPU test FAIL: {e}")