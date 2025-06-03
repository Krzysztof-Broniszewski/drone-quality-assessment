import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import pyiqa

# === Ustawienia ===
frames_dir = "data/frames"
output_csv = "data/frames_quality_scores.csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Modele IQA === (ładują się tylko raz)
metrics = {
    "brisque": pyiqa.create_metric('brisque').to(device),
    "niqe": pyiqa.create_metric('niqe').to(device),
    "clipiqa": pyiqa.create_metric('clipiqa').to(device),
    "maniqa": pyiqa.create_metric('maniqa').to(device)
}

# === Transformacja wejściowa ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Przetwarzanie plików ===
results = []

for filename in os.listdir(frames_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(frames_dir, filename)

        try:
            img = Image.open(full_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            variance = tensor.var().item()
            if variance < 1e-5:
                print(f"⚠️ Pominięto {filename} – zbyt niska wariancja pikseli")
                continue

            entry = {"filename": filename}

            with torch.no_grad():
                for name, model in metrics.items():
                    score = model(tensor).item()
                    entry[name] = round(score, 3)

            results.append(entry)
            print(f"✅ {filename} → {entry}")

        except Exception as e:
            print(f"❌ Błąd przy {filename}: {e}")

# === Zapis do CSV ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n📄 Zapisano wyniki do {output_csv}")
