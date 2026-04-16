
import torch

print(f"CUDA Available? {torch.cuda.is_available()}")
print(f"CUDA driver version: {torch.version.cuda}")
print(f"CUDNN version: {torch.backends.cudnn.version()}")
print(f"Device count: {torch.cuda.device_count()}")


