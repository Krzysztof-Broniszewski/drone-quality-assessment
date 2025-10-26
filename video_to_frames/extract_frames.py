# video_to_frames/extract_frames.py
from pathlib import Path
import argparse, csv, cv2

# --- FIX: dołącz root projektu do sys.path ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]  # .../drone-quality-assessment
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ---------------------------------------------

from tools.naming import build_frame_name, slugify

def extract_frames_with_time(video_path: Path, out_root: Path, step_sec: float = 5.0) -> Path:
    video_path = Path(video_path).resolve()
    out_dir = out_root / slugify(video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie mogę otworzyć: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_ms = (frames / fps * 1000.0) if (fps>0 and frames>0) else 0.0

    print(f"🎞️ {video_path.name} | krok={step_sec}s | FPS={fps:.2f} | klatek={frames} | szac.czas={duration_ms/1000:.2f}s")

    t_ms = 0.0
    rows = []
    saved = 0
    while duration_ms == 0.0 or t_ms <= duration_ms + 1.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ok, frame = cap.read()
        if not ok:
            break
        fname = build_frame_name(video_path, t_ms, ".jpg")
        (out_dir / fname).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / fname), frame)
        rows.append({"name": fname, "timestamp_ms": int(round(t_ms))})
        saved += 1
        t_ms += step_sec * 1000.0

    cap.release()

    # index CSV w folderze klatek
    index_csv = out_dir / "frames_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name","timestamp_ms"])
        w.writeheader(); w.writerows(rows)

    print(f"✅ Zapisano {saved} klatek → {out_dir}")
    print(f"📄 Index: {index_csv}")
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Ścieżka do pliku .mp4")
    ap.add_argument("--out", default="data/frames", help="Folder wyjściowy")
    ap.add_argument("--step", type=float, default=5.0, help="Krok w sekundach, np. 5.0")
    args = ap.parse_args()
    extract_frames_with_time(Path(args.video), Path(args.out), args.step)

if __name__ == "__main__":
    main()

