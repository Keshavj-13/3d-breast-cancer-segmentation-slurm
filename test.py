import torch

print("Torch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

