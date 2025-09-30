import argparse
import io
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

try:
    from PIL import Image
except Exception:
    Image = None

# TODO: (@gas) convert to tests

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
BASE = f"http://{HOST}:{PORT}"

USER = os.getenv("APP_USERNAME", "admin")
PASS = os.getenv("APP_PASSWORD", "secret")

# ------------------------
# Image helpers
# ------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def collect_files_from_arg(entry: str) -> List[Path]:
    """
    Accept either a file, a directory, or a comma-separated list of both.
    Expand into a flat list of image files (non-recursive for simplicity).
    """
    paths: List[Path] = []
    for token in entry.split(","):
        token = token.strip()
        if not token:
            continue
        p = Path(token)
        if p.is_dir():
            paths.extend([x for x in p.iterdir() if is_image_file(x)])
        elif is_image_file(p):
            paths.append(p)
        else:
            print(f"Warning: {p} is not a valid image file/dir; skipping", file=sys.stderr)
    return paths


def collect_pos_from_mapping_args(pos_args: List[str]) -> Dict[str, List[Path]]:
    """
    Parse repeated --pos CLASS=path1[,path2,...]
    Return {class_id: [Path, ...]}
    """
    result: Dict[str, List[Path]] = {}
    for spec in pos_args:
        if "=" not in spec:
            print(f"Warning: --pos expects CLASS=paths, got: {spec}", file=sys.stderr)
            continue
        cls, paths_str = spec.split("=", 1)
        cls = cls.strip()
        files = collect_files_from_arg(paths_str)
        if files:
            result.setdefault(cls, []).extend(files)
    return result


def collect_pos_from_root_dir(root_dir: str) -> Dict[str, List[Path]]:
    """
    Parse --pos-dir ROOT where ROOT/<class_id>/*.png|jpg...
    """
    result: Dict[str, List[Path]] = {}
    if not root_dir:
        return result
    root = Path(root_dir)
    if not root.is_dir():
        print(f"Warning: --pos-dir {root} is not a directory", file=sys.stderr)
        return result
    for sub in sorted([d for d in root.iterdir()]):
        if sub.is_dir():
            cls = sub.name
            files = [x for x in sub.iterdir() if is_image_file(x)]
            if files:
                result[cls] = files
        elif is_image_file(sub):
            if "0" not in result:
                result["0"] = []
            result["0"].append(sub)
    return result


def read_image_bytes(p: Path) -> bytes:
    """
    Always send valid PNG bytes if Pillow is available; otherwise raw bytes.
    """
    if Image is None:
        return p.read_bytes()
    try:
        im = Image.open(str(p)).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # fallback to raw
        return p.read_bytes()


# ------------------------
# Random image fallback
# ------------------------

def make_random_image_bytes(size=(256, 256)) -> bytes:
    if Image is not None:
        import random
        img = Image.new("RGB", size, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    # fallback raw
    return os.urandom(size[0] * size[1] // 2)


# ------------------------
# Multipart builders
# ------------------------

def files_for_set_payload(mapping: Dict[str, List[bytes]], negatives: List[bytes]) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    """
    Build multipart for /api/v1/set:
      positive[<class_id>]  (repeat per image)
      negative              (repeat per image; optional)
    """
    files = []
    for class_id, imgs in mapping.items():
        field = f"positive[{class_id}]"
        for idx, content in enumerate(imgs):
            files.append((field, (f"{class_id}_{idx}.png", content, "image/png")))
    for idx, content in enumerate(negatives or []):
        files.append(("negative", (f"neg_{idx}.png", content, "image/png")))
    return files


def files_list(field: str, img_bytes_list: List[bytes]) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    out = []
    for i, content in enumerate(img_bytes_list):
        out.append((field, (f"img_{i}.png", content, "image/png")))
    return out


# ------------------------
# Main flow
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="Test client for detection FastAPI service.")
    parser.add_argument("--host", default=HOST, help="Server host (default from HOST env)")
    parser.add_argument("--port", type=int, default=PORT, help="Server port (default from PORT env)")

    # Positives:
    parser.add_argument("--pos", action="append", default=[],
                        help="Repeatable. CLASS=path1[,path2,...] (files or dirs).")
    parser.add_argument("--pos-dir", default="",
                        help="Root dir where each subfolder is a class (ROOT/<class>/*.png).")

    # Negatives:
    parser.add_argument("--neg", default="",
                        help="Comma-separated files/dirs for negatives.")
    parser.add_argument("--neg-dir", default="",
                        help="Directory of negative images (flat).")

    # Infer batch:
    parser.add_argument("--infer", default="",
                        help="Comma-separated files/dirs for /infer images. If empty, randoms will be generated.")
    parser.add_argument("--infer-count", type=int, default=3,
                        help="If --infer empty, how many random images to send (default 3).")

    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    s = requests.Session()
    s.auth = (USER, PASS)

    # Health
    r = s.get(f"{base}/health")
    print("GET /health:", r.status_code, r.json())

    # -------- Build /set payload from inputs --------
    pos_map: Dict[str, List[Path]] = {}
    # From --pos CLASS=paths
    pos_map_arg = collect_pos_from_mapping_args(args.pos)
    for k, v in pos_map_arg.items():
        pos_map.setdefault(k, []).extend(v)
    # From --pos-dir ROOT/<class>/*
    pos_map_dir = collect_pos_from_root_dir(args.pos_dir)
    for k, v in pos_map_dir.items():
        pos_map.setdefault(k, []).extend(v)

    neg_files: List[Path] = []
    if args.neg:
        neg_files.extend(collect_files_from_arg(args.neg))
    if args.neg_dir:
        neg_files.extend(collect_files_from_arg(args.neg_dir))

    # Convert to bytes
    pos_bytes_map: Dict[str, List[bytes]] = {cls: [read_image_bytes(p) for p in files] for cls, files in pos_map.items()}
    neg_bytes: List[bytes] = [read_image_bytes(p) for p in neg_files]

    # Only call /set if anything was provided; otherwise skip (detector may still work)
    if pos_bytes_map or neg_bytes:
        set_files = files_for_set_payload(pos_bytes_map, neg_bytes)
        r = s.post(f"{base}/api/v1/set", files=set_files)
        print("POST /api/v1/set:", r.status_code, r.headers.get("X-Server-Latency-ms"))
        if r.status_code != 204:
            print("Error /api/v1/set:", r.text, file=sys.stderr)
            sys.exit(2)
    else:
        print("No positives/negatives provided; skipping /api/v1/set.")

    # -------- Build /infer payload --------
    infer_files: List[Path] = []
    if args.infer:
        infer_files.extend(collect_files_from_arg(args.infer))

    if infer_files:
        infer_bytes = [read_image_bytes(p) for p in infer_files]
    else:
        # fallback to random
        infer_bytes = [make_random_image_bytes() for _ in range(max(1, args.infer_count))]

    infer_payload = files_list("images", infer_bytes)
    r = s.post(f"{base}/api/v1/infer", files=infer_payload)
    print("POST /api/v1/infer:", r.status_code, "latency(ms)=", r.headers.get("X-Server-Latency-ms"))
    try:
        batch = r.json()
    except Exception:
        print("Non-JSON response:", r.text)
        sys.exit(3)

    # Expect a list of per-image results
    if isinstance(batch, list):
        print("Batch length:", len(batch))
        print("Detections:", batch)
    else:
        print("Unexpected response type:", type(batch), batch)


if __name__ == "__main__":
    main()
