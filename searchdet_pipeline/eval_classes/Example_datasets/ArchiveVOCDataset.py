from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from searchdet_pipeline.eval_classes.Dataset import Dataset
from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation


class ArchiveVOCDataset(Dataset):
    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "ArchiveVOCDataset":
        annotations = cls._build_annotations(obj, base_dir=None)
        meta = obj.get("meta", {})
        meta.update(
            {
                "dataset_type": "voc_detection",
                "total_images": len(obj.get("images", [])),
                "total_annotations": len(obj.get("annotations", [])),
                "categories": obj.get("categories", []),
            }
        )
        data = DatasetModel(data_points=annotations, meta=meta)
        return cls(dataset=data)

    @classmethod
    def from_path(cls, path: Path) -> "ArchiveVOCDataset":
        base = Path(path)
        if base.is_dir() and base.name == "annotations":
            base_dir = base.parent
            ann_dir = base
        else:
            base_dir = base
            ann_dir = base_dir / "annotations"
        img_dir = base_dir / "images"
        xml_files = sorted(list(ann_dir.glob("*.xml")))
        anns: List[COCOAnnotation] = []
        categories: List[str] = []
        for xml_path in xml_files:
            try:
                file_anns, cats = cls._parse_single_voc_xml(xml_path, img_dir)
                anns.extend(file_anns)
                categories.extend(cats)
            except Exception:
                continue
        unique_categories = sorted(set(categories))
        meta: Dict[str, Any] = {
            "dataset_type": "voc_detection",
            "source_path": str(base_dir),
            "base_directory": str(base_dir),
            "total_images": len({a.file_name for a in anns}),
            "total_annotations": len(anns),
            "categories": unique_categories,
            "name": "archive_voc",
        }
        data = DatasetModel(data_points=anns, meta=meta)
        return cls(dataset=data)

    @classmethod
    def _build_annotations(
        cls, obj: dict[str, Any], base_dir: Path | None
    ) -> List[COCOAnnotation]:
        images_by_id: Dict[int, Dict[str, Any]] = {
            int(im.get("id")): im for im in obj.get("images", []) if "id" in im
        }
        categories_by_id: Dict[int, Dict[str, Any]] = {
            int(cat.get("id")): cat for cat in obj.get("categories", []) if "id" in cat
        }
        anns: List[COCOAnnotation] = []
        for ann in obj.get("annotations", []) or []:
            try:
                image_id = int(ann.get("image_id"))
                image_info = images_by_id.get(image_id)
                if not image_info:
                    continue

                file_name: str = image_info.get("file_name", "")
                width: int = int(image_info.get("width", 0) or 0)
                height: int = int(image_info.get("height", 0) or 0)
                cat_id = ann.get("category_id")
                cat = categories_by_id.get(int(cat_id)) if cat_id is not None else None
                label: int | str = (
                    cat.get("name")
                    if isinstance(cat, dict) and "name" in cat
                    else (int(cat_id) if cat_id is not None else -1)
                )
                bbox_list = ann.get("bbox") or []
                bbox: List[float] = [float(v) for v in bbox_list] if isinstance(
                    bbox_list, (list, tuple)
                ) else []
                if len(bbox) == 4:
                    x, y, w, h = bbox
                else:
                    x = y = 0.0
                    w = float(width)
                    h = float(height)
                    bbox = [x, y, w, h]
                area_val = ann.get("area")
                if area_val is None:
                    area = float(w * h)
                else:
                    try:
                        area = float(area_val)
                    except Exception:
                        area = float(w * h)
                mask_np = np.zeros((height, width), dtype=np.uint8)
                if mask_np.sum() == 0 and len(bbox) == 4:
                    x0, y0, bw, bh = bbox
                    x1, y1 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
                    x2 = int(min(width, np.ceil(x0 + bw)))
                    y2 = int(min(height, np.ceil(y0 + bh)))
                    if x2 > x1 and y2 > y1:
                        mask_np[y1:y2, x1:x2] = 1

                if base_dir is not None and file_name:
                    img_path = base_dir / file_name
                    if img_path.exists():
                        try:
                            img_arr = np.array(Image.open(img_path).convert("RGB"))
                        except Exception:
                            img_arr = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        img_arr = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    img_arr = np.zeros((height, width, 3), dtype=np.uint8)

                anns.append(
                    COCOAnnotation(
                        img=img_arr,
                        mask=mask_np,
                        label=label,
                        image_size=(int(width), int(height)),
                        width=int(width),
                        height=int(height),
                        area=float(area),
                        file_name=file_name,
                        bbox=bbox,
                    )
                )
            except Exception:
                continue
        return anns

    @staticmethod
    def _parse_single_voc_xml(xml_path: Path, img_dir: Path) -> Tuple[List[COCOAnnotation], List[str]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_el = root.find("filename")
        file_name = filename_el.text.strip() if filename_el is not None and filename_el.text else ""
        size_el = root.find("size")
        width = int(size_el.findtext("width", default="0")) if size_el is not None else 0
        height = int(size_el.findtext("height", default="0")) if size_el is not None else 0

        img_path = img_dir / file_name if file_name else None
        if img_path and img_path.exists():
            try:
                img_arr = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                img_arr = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            img_arr = np.zeros((height, width, 3), dtype=np.uint8)

        anns: List[COCOAnnotation] = []
        cats: List[str] = []

        for obj in root.findall("object"):
            name_el = obj.find("name")
            label = name_el.text.strip() if name_el is not None and name_el.text else "object"
            cats.append(label)

            bnd = obj.find("bndbox")
            if bnd is not None:
                try:
                    xmin = float(bnd.findtext("xmin", default="0"))
                    ymin = float(bnd.findtext("ymin", default="0"))
                    xmax = float(bnd.findtext("xmax", default="0"))
                    ymax = float(bnd.findtext("ymax", default="0"))
                except Exception:
                    xmin = ymin = 0.0
                    xmax = float(width)
                    ymax = float(height)
            else:
                xmin = ymin = 0.0
                xmax = float(width)
                ymax = float(height)

            # трансформация VOC bbox -> COCO bbox 
            x = max(0.0, xmin)
            y = max(0.0, ymin)
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            bbox = [x, y, w, h]
            area = float(w * h)
            mask_np = np.zeros((height, width), dtype=np.uint8)
            x1, y1 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
            x2 = int(min(width, np.ceil(x + w)))
            y2 = int(min(height, np.ceil(y + h)))
            if x2 > x1 and y2 > y1:
                mask_np[y1:y2, x1:x2] = 1

            anns.append(
                COCOAnnotation(
                    img=img_arr,
                    mask=mask_np,
                    label=label,
                    image_size=(int(width), int(height)),
                    width=int(width),
                    height=int(height),
                    area=area,
                    file_name=file_name,
                    bbox=bbox,
                )
            )

        return anns, cats