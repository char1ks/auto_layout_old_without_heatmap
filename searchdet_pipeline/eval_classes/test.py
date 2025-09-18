from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional
import xml.etree.ElementTree as ET
try:
    from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset


ImageDir = Optional[Path]

def _auto_resolve_ann_dir(dataset_dir: Path, ann_dir: Optional[Path]) -> Tuple[Optional[Path], List[Path]]:
    candidates: List[Path] = []
    if ann_dir is not None:
        candidates.append(ann_dir)
    for name in [
        "annotations", "Annotations", "annotation", "Annotation",
        "labels", "Labels", "xml", "XML",
        "VOC2007/Annotations", "VOC2012/Annotations",
    ]:
        candidates.append(dataset_dir / name)

    for c in candidates:
        if c.exists() and c.is_dir():
            xmls = _collect_xmls(c)
            if xmls:
                return c, xmls

    xmls = _collect_xmls(dataset_dir)
    if xmls:
        return xmls[0].parent, xmls

    return ann_dir, []


def _collect_xmls(base_dir: Path) -> List[Path]:
    return sorted(p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")


def _find_image_dirs(base_dir: Path) -> List[Path]:
    candidates = [
        base_dir / "images",
        base_dir / "Images",
        base_dir / "image",
        base_dir / "Image",
        base_dir / "JPEGImages",
        base_dir / "jpegimages",
        base_dir / "JPEGIMAGES",
        base_dir,
    ]
    return [d for d in candidates if d.exists() and d.is_dir()]


def _count_images_in_dir(img_dir: Path) -> int:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts)


def _extract_filename_from_xml(xml_path: Path) -> str:
    try:
        root = ET.parse(xml_path).getroot()
        fn = root.findtext("filename", default="")
        return (fn or "").strip()
    except Exception:
        return ""


def _resolve_img_path(file_name: str, img_dirs: List[Path]) -> Optional[Path]:
    if not file_name:
        return None
    fn_path = Path(file_name)
    if fn_path.is_absolute():
        return fn_path if fn_path.exists() else None
    for d in img_dirs:
        cand = d / file_name
        if cand.exists():
            return cand
        alt = d.parent / file_name
        if alt.exists():
            return alt
    return None


def main() -> int:
    args = sys.argv[1:]
    dataset_dir = Path("archive")
    ann_dir: Optional[Path] = None
    img_dir: Optional[Path] = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--dataset", "--root", "--data"):
            if i + 1 < len(args):
                dataset_dir = Path(args[i + 1])
                i += 2
                continue
        if arg in ("--ann", "--ann-dir", "--annotations"):
            if i + 1 < len(args):
                ann_dir = Path(args[i + 1])
                i += 2
                continue
        if arg in ("--img", "--img-dir", "--images"):
            if i + 1 < len(args):
                img_dir = Path(args[i + 1])
                i += 2
                continue
        if not arg.startswith("-") and str(dataset_dir) == "archive":
            dataset_dir = Path(arg)
            i += 1
            continue
        i += 1

    if not dataset_dir.exists():
        return 1

    resolved_ann_dir, xml_files = _auto_resolve_ann_dir(dataset_dir, ann_dir)
    img_dirs = [img_dir] if img_dir is not None else _find_image_dirs(dataset_dir)

    try:
        dataset = ArchiveVOCDataset.from_path(dataset_dir, ann_dir=resolved_ann_dir, img_dir=img_dir)
    except Exception:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())