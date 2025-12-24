import torch
import torch.nn as nn
import os

# Force the override inside the script to be sure
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda")

try:
    print("\n1. Testing Basic Tensor Allocation...")
    x = torch.randn(16, 69, 256).to(device)
    print("   ✅ Success")

    print("\n2. Testing Linear Layer...")
    l = nn.Linear(256, 256).to(device)
    y = l(x)
    print("   ✅ Success")

    print("\n3. Testing Scaled Dot Product Attention (Math Path)...")
    # Force 'Math' path which is slow but stable
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        t = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True).to(device)
        z = t(x)
    print("   ✅ Success (Math Path)")

    print("\n4. Testing Default Attention (Likely Crash Point)...")
    # This uses the default backend (Flash/MemEfficient)
    t2 = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True).to(device)
    z2 = t2(x)
    print("   ✅ Success (Default Path)")

except Exception as e:
    print(f"\n❌ Python Error: {e}")