# tools/naming.py
from pathlib import Path
import re

_slug_re = re.compile(r"[^A-Za-z0-9._-]+")

def slugify(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return _slug_re.sub("", name)

def hmsms_from_ms(ms: float):
    total_ms = int(round(ms))
    s, ms = divmod(total_ms, 1000)
    m, s  = divmod(s, 60)
    h, m  = divmod(m, 60)
    return h, m, s, ms

def build_frame_name(video_path: str | Path, t_ms: float, ext: str = ".jpg") -> str:
    stem = slugify(Path(video_path).stem)
    h, m, s, ms = hmsms_from_ms(t_ms)
    return f"{stem}_{h:02d}h{m:02d}m{s:02d}s{ms:03d}ms{ext}"
