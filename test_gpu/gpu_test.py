import torch

if torch.cuda.is_available():
    print("✅ GPU dostępne:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU niedostępne. Sprawdź instalację CUDA lub sterowniki.")
