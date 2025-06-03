import os
import torch
from PIL import Image
from torchvision import transforms
from model.cnn_model import MultiHeadCNN

# === Ścieżki ===
frames_dir = "data/frames"
model_path = "model/model_weights.pth"  # <- Ścieżka do wag modelu

# === Transformacje (224x224) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Wczytanie modelu ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadCNN(num_classes=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Przetwarzanie klatek ===
for filename in os.listdir(frames_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(frames_dir, filename)

        # Wczytaj obraz
        img = Image.open(full_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # Predykcja
        with torch.no_grad():
            outputs = model(img_tensor)

        # Dla każdej cechy: wybieramy klasę z największym prawdopodobieństwem
        preds = {
            key: torch.argmax(outputs[key], dim=1).item() + 1  # +1 bo klasy 0–4 → zapis 1–5
            for key in outputs
        }

        # Nowa nazwa
        base, ext = os.path.splitext(filename)
        new_name = f"{base}__q{preds['ostrosc']}_s{preds['swiatlo']}_e{preds['ekspozycja']}_k{preds['kadr']}{ext}"
        new_path = os.path.join(frames_dir, new_name)

        os.rename(full_path, new_path)
        print(f"✅ {filename} → {new_name}")
