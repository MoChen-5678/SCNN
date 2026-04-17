import torch
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前设备:", torch.cuda.get_device_name(0))