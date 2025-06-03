import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, interval_sec=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎞️ Przetwarzanie: {video_path}")
    print(f"🎯 Klatka co {interval_sec} sekund (co {frame_interval} klatek)")

    count = 0
    saved = 0
    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved += 1
        count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"✅ Zapisano {saved} klatek do: {output_dir}")

# Przykład użycia (można zmienić na argparse lub GUI w przyszłości)
if __name__ == "__main__":
    video_file = "data/videos/Tower in Muszyna.MP4.mp4"  # <- Podmień nazwę
    output_folder = "data/frames"
    extract_frames(video_file, output_folder, interval_sec=5)
