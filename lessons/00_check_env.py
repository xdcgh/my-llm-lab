import platform
import sys

import torch

print("Python version:", sys.version.split()[0])
print("Python executable:", sys.executable)
print("Python machine:", platform.machine())
print("PyTorch version:", torch.__version__)
print("MPS is_built:", torch.backends.mps.is_built())
print("MPS is_available:", torch.backends.mps.is_available())

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available. Please ensure you are running on a Mac with an Apple Silicon GPU and that you have the correct version of PyTorch installed."
    )

for device in ["cpu", "mps"]:
    x = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True
    )
    value = (x * x).sum()
    value.backward()
    print(f"{device}: value: {value.item()}, gradient: {x.grad.cpu().tolist()}")
