import torch
import os

if os.name == 'nt':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
elif os.name == 'posix':
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
else:
    print("Unsupported operating")

print(f"device: {device}")