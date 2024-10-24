import torch
print(torch.version.cuda)  # Should print 12.4 if PyTorch is using CUDA 12.4
print(torch.cuda.is_available())  # Should print True if GPU is accessible