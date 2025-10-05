from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple
import xml.etree.ElementTree as ET
import sys
import numpy as np
from PIL import Image
from evaluate.dataset import Dataset
from evaluate.dataset_model import DatasetModel
from evaluate.dataset_meta import DatasetMeta
from evaluate.coco_annotation import CocoAnnotation
from evaluate.logging import get_logger
logger = get_logger(__name__)

class VocDataset(Dataset):
    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "VocDataset":
        annotations = cls._build_annotations(obj, base_dir=None)
        existing_meta = obj.get("meta", {})
        
        meta = DatasetMeta(
            dataset_type="voc_detection",
            total_images=len(obj.get("images", [])),
            total_annotations=len(obj.get("annotations", [])),
            categories=obj.get("categories", []),
            uid=existing_meta.get("uid"),
            name=existing_meta.get("name"),
            url=existing_meta.get("url"),
            color_channels=existing_meta.get("color_channels", []),
            source_path=existing_meta.get("source_path"),
            base_directory=existing_meta.get("base_directory"),
            extra=existing_meta.get("extra", {})
        )
        data = DatasetModel(data_points=annotations, meta=meta)
        return cls(dataset=data)

    @classmethod
    def from_path(cls, path: Path, **kwargs: Any) -> "VocDataset":
        ann_dir = kwargs.get('ann_dir')
        img_dir = kwargs.get('img_dir')
        base_dir = Path(path)

        xml_files = []
        if ann_dir:
            xml_files = sorted(p for p in Path(ann_dir).rglob("*.xml") if p.is_file())

        chosen_img_dir = Path(img_dir) if img_dir else base_dir

        anns, categories = [], []
        for xml_path in xml_files:
            try:
                file_anns, cats = cls._parse_single_voc_xml(xml_path, chosen_img_dir)
                anns.extend(file_anns)
                categories.extend(cats)
            except Exception as e:
                logger.warning(f"Не удалось обработать XML файл {xml_path}: {e}")
                continue

        meta = DatasetMeta(
            dataset_type="voc_detection",
            source_path=str(base_dir),
            base_directory=str(base_dir),
            total_images=len({a.file_name for a in anns}),
            total_annotations=len(anns),
            categories=sorted(set(categories)),
            name="archive_voc",
        )
        return cls(dataset=DatasetModel(data_points=anns, meta=meta))

    @staticmethod
    def _build_annotations(obj: dict[str, Any], base_dir: Path | None) -> List[CocoAnnotation]:
        images_by_id = {int(im["id"]): im for im in obj.get("images", []) if "id" in im}
        categories_by_id = {int(cat["id"]): cat for cat in obj.get("categories", []) if "id" in cat}
        anns = []
        
        for ann in obj.get("annotations", []):
            try:
                image_id = int(ann.get("image_id"))
                image_info = images_by_id.get(image_id)
                if not image_info:
                    continue

                file_name = image_info.get("file_name", "")
                width, height = int(image_info.get("width", 0) or 0), int(image_info.get("height", 0) or 0)
                
                cat_id = ann.get("category_id")
                cat = categories_by_id.get(int(cat_id)) if cat_id is not None else None
                label = cat.get("name") if cat and "name" in cat else (int(cat_id) if cat_id is not None else -1)
                
                bbox_list = ann.get("bbox", [])
                bbox = [float(v) for v in bbox_list] if isinstance(bbox_list, (list, tuple)) and len(bbox_list) == 4 else [0.0, 0.0, float(width), float(height)]
                x, y, w, h = bbox
                
                area = float(ann.get("area", w * h))
                mask_np = np.zeros((height, width), dtype=np.uint8)
                if len(bbox) == 4:
                    x1, y1 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
                    x2, y2 = int(min(width, np.ceil(x + w))), int(min(height, np.ceil(y + h)))
                    if x2 > x1 and y2 > y1:
                        mask_np[y1:y2, x1:x2] = 1
                img_arr: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
                if base_dir and file_name:
                    img_path = base_dir / file_name
                    try:
                        img_arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                    except Exception:
                        pass
                anns.append(CocoAnnotation(
                    img=img_arr, mask=mask_np, label=label, image_size=(width, height),
                    width=width, height=height, area=area, file_name=file_name, bbox=bbox
                ))
            except Exception:
                continue
        return anns

    @staticmethod
    def _parse_single_voc_xml(xml_path: Path, img_dir: Path) -> Tuple[List[CocoAnnotation], List[str]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        file_name = (root.findtext("filename") or "").strip()
        xml_w, xml_h = int(root.findtext("size/width", "0") or 0), int(root.findtext("size/height", "0") or 0)

        img_path = None
        if file_name:
            p = Path(file_name)
            img_path = p if p.is_absolute() else img_dir / file_name

        if img_path:
            try:
                img_arr = np.array(Image.open(img_path).convert("RGB"))
                H_img, W_img = img_arr.shape[:2]
            except Exception:
                W_img, H_img = xml_w, xml_h
                img_arr = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        else:
            W_img, H_img = xml_w, xml_h
            img_arr = np.zeros((H_img, W_img, 3), dtype=np.uint8)

        swapped = xml_w > 0 and xml_h > 0 and xml_w == H_img and xml_h == W_img

        def voc_xyxy_to_xywh_inclusive(xmin, ymin, xmax, ymax, W, H):
            x, y = max(0.0, float(xmin) - 1.0), max(0.0, float(ymin) - 1.0)
            w, h = max(0.0, float(xmax) - float(xmin) + 1.0), max(0.0, float(ymax) - float(ymin) + 1.0)
            if W > 0 and H > 0:
                x, y = max(0.0, min(x, W - 1.0)), max(0.0, min(y, H - 1.0))
                w, h = max(0.0, min(w, W - x)), max(0.0, min(h, H - y))
            return x, y, w, h

        def rotate90_ccw_xywh(x, y, w, h, Wsrc):
            x0, y0, x1, y1 = x, y, x + w - 1.0, y + h - 1.0
            pts = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)
            xr, yr = pts[:, 1], (Wsrc - 1.0) - pts[:, 0]
            xr0, yr0, xr1, yr1 = xr.min(), yr.min(), xr.max(), yr.max()
            return xr0, yr0, xr1 - xr0 + 1.0, yr1 - yr0 + 1.0

        anns, cats = [], []
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "object").strip()
            cats.append(label)
            
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
                
            xmin, ymin = float(bnd.findtext("xmin", "0")), float(bnd.findtext("ymin", "0"))
            xmax, ymax = float(bnd.findtext("xmax", str(xml_w or W_img))), float(bnd.findtext("ymax", str(xml_h or H_img)))

            if swapped and xml_w > 0 and xml_h > 0:
                x, y, w, h = voc_xyxy_to_xywh_inclusive(xmin, ymin, xmax, ymax, xml_w, xml_h)
                x, y, w, h = rotate90_ccw_xywh(x, y, w, h, xml_w)
                x, y = max(0.0, min(x, W_img - 1.0)), max(0.0, min(y, H_img - 1.0))
                w, h = max(0.0, min(w, W_img - x)), max(0.0, min(h, H_img - y))
            else:
                x, y, w, h = voc_xyxy_to_xywh_inclusive(xmin, ymin, xmax, ymax, W_img, H_img)

            area = float(w * h)
            mask_np = np.zeros((H_img, W_img), dtype=np.uint8)
            x1, y1 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
            x2, y2 = int(min(W_img, np.ceil(x + w))), int(min(H_img, np.ceil(y + h)))
            if x2 > x1 and y2 > y1:
                mask_np[y1:y2, x1:x2] = 1

            anns.append(CocoAnnotation(
                img=img_arr, mask=mask_np, label=label, image_size=(W_img, H_img),
                width=W_img, height=H_img, area=area, file_name=file_name, bbox=[x, y, w, h]
            ))

        return anns, cats