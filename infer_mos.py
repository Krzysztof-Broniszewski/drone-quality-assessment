# infer_mos.py
# Użycie:
#   python infer_mos.py --images data/frames/Tower_in_Muszyna_MP4 --weights best_mos.pth --out results/Tower_in_Muszyna_MP4/drone_mos_predictions.csv
#   opcjonalnie: --device cpu  oraz  --rename (dopisać __mos do nazw plików)

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from torchvision import transforms
import torchvision.models as models

class CNNforMOS(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.mos = nn.Linear(in_features, 1)
    def forward(self, x):
        f = self.backbone(x)
        return self.mos(f)

def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def predict_folder(img_dir: Path, weights: Path, out_csv: Path, device_str="cuda", rename=False):
    device = torch.device("cuda" if (device_str=="cuda" and torch.cuda.is_available()) else "cpu")
    model = CNNforMOS().to(device).eval()
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)

    tfm = build_transform()
    img_dir = Path(img_dir)
    files = sorted([*img_dir.glob("*.jpg"), *img_dir.glob("*.jpeg"), *img_dir.glob("*.png")])

    rows = []
    with torch.inference_mode():
        for p in files:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = tfm(img).unsqueeze(0).to(device)
            mos = model(x).item()
            mos = max(0.0, min(100.0, mos))  # clip 0–100
            rows.append({"name": p.name, "mos_pred": round(mos, 3)})

    df = pd.DataFrame(rows).sort_values("mos_pred", ascending=False).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Zapisano: {out_csv}  |  liczba klatek: {len(df)}")

    if rename:
        # dopisz __mosXX.X do nazw plików (bez nadpisywania)
        for _, row in df.iterrows():
            src = img_dir / row["name"]
            if not src.exists():
                continue
            stem, ext = src.stem, src.suffix
            if "__mos" in stem:
                continue
            dst = img_dir / f"{stem}__mos{row['mos_pred']:.1f}{ext}"
            i = 1
            while dst.exists():
                dst = img_dir / f"{stem}__mos{row['mos_pred']:.1f}_{i}{ext}"
                i += 1
            src.rename(dst)
        print("Dopisano __mos do nazw plików.")

    return df

def default_out(images_dir: Path) -> Path:
    slug = images_dir.name
    return Path("results") / slug / "drone_mos_predictions.csv"

def main():
    ap = argparse.ArgumentParser(description="Inferencja MOS (0–100) dla folderu klatek.")
    ap.add_argument("--images", required=True, help="Folder z klatkami (np. data/frames/<film>)")
    ap.add_argument("--weights", default="best_mos.pth", help="Ścieżka do best_mos.pth")
    ap.add_argument("--out", default="", help="Ścieżka do wyniku CSV (domyślnie results/<folder>/drone_mos_predictions.csv)")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="Urządzenie")
    ap.add_argument("--rename", action="store_true", help="Dopisz __mosXX.X do nazw plików")
    args = ap.parse_args()

    images_dir = Path(args.images)
    out_csv = Path(args.out) if args.out else default_out(images_dir)
    predict_folder(images_dir, Path(args.weights), out_csv, device_str=args.device, rename=args.rename)

if __name__ == "__main__":
    main()
