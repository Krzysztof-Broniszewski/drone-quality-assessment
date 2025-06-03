import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pyiqa

# === Konfiguracja modelu BRISQUE ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = pyiqa.create_metric('brisque').to(device)

# === Ścieżka do folderu ===
frames_dir = "data/frames"

# === Transformacja obrazu ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Przetwarzanie obrazów ===
for filename in os.listdir(frames_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(frames_dir, filename)

        # Wczytaj obraz i przekształć
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            score = model(img_tensor).item()

        score_rounded = round(score)
        base, ext = os.path.splitext(filename)
        new_name = f"{base}__brisque{score_rounded}{ext}"
        new_path = os.path.join(frames_dir, new_name)

        os.rename(path, new_path)
        print(f"✅ {filename} → {new_name}")

