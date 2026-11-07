
import sys
import platform
import torch
import torchvision
import datasets

print("=" * 45)
print("LiverPath AI - Environment Check")
print("=" * 45)

print("Python:", sys.version.split()[0])
print("System:", platform.platform())
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Hugging Face datasets:", datasets.__version__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Compute device:", device)

tensor = torch.randn(4, 3, 224, 224, device=device)
print("Tensor shape:", tensor.shape)

print("\nEnvironment check successful!")
