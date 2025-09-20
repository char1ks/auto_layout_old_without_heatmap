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
    def from_path(cls, path: Path, ann_dir: Optional[Path] = None, img_dir: Optional[Path] = None) -> "ArchiveVOCDataset":
        base = Path(path)
        # Определяем базовую директорию датасета (корень)
        if base.is_dir() and base.name.lower() in {"annotations", "annotation"}:
            base_dir = base.parent
        else:
            base_dir = base

        # 1) Сбор XML
        xml_files: List[Path] = []
        if ann_dir is not None:
            ann_dir = Path(ann_dir)
            if ann_dir.exists() and ann_dir.is_dir():
                xml_files.extend(sorted(p for p in ann_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"))
        else:
            # Старое поведение: известные папки аннотаций рядом с корнем
            candidate_ann_dirs = [
                base_dir / "annotations",
                base_dir / "Annotations",
                base_dir / "annotation",
                base_dir / "Annotation",
            ]
            for d in candidate_ann_dirs:
                if d.exists() and d.is_dir():
                    xml_files.extend(sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"))
            # Если ничего не нашли, пробуем рекурсивно по всей базе (на случай кастомной структуры)
            if not xml_files:
                xml_files = sorted(p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")

        # 2) Определяем папку с изображениями
        if img_dir is not None:
            img_dir = Path(img_dir)
            chosen_img_dir = img_dir if img_dir.exists() and img_dir.is_dir() else base_dir
        else:
            img_dir_candidates = [
                base_dir / "images",
                base_dir / "Images",
                base_dir / "image",
                base_dir / "Image",
                base_dir / "JPEGImages",
                base_dir / "jpegimages",
                base_dir / "JPEGIMAGES",
                base_dir,
            ]
            chosen_img_dir = next((d for d in img_dir_candidates if d.exists() and d.is_dir()), base_dir)

        # 3) Парсим XML
        anns: List[COCOAnnotation] = []
        categories: List[str] = []
        for xml_path in xml_files:
            try:
                file_anns, cats = cls._parse_single_voc_xml(xml_path, chosen_img_dir)
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
        import numpy as np
        tree = ET.parse(xml_path)
        root = tree.getroot()

        file_name = (root.findtext("filename") or "").strip()
        xml_w = int(root.findtext("size/width",  "0") or 0)
        xml_h = int(root.findtext("size/height", "0") or 0)

        # Найдём картинку и возьмём РЕАЛЬНЫЕ размеры
        img_path: Optional[Path] = None
        if file_name:
            p = Path(file_name)
            if p.is_absolute():
                img_path = p
            else:
                cand = img_dir / file_name
                img_path = cand if cand.exists() else (img_dir.parent / file_name)

        if img_path and img_path.exists():
            try:
                img_arr = np.array(Image.open(img_path).convert("RGB"))
                H_img, W_img = img_arr.shape[0], img_arr.shape[1]
            except Exception:
                W_img, H_img = xml_w, xml_h
                img_arr = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        else:
            W_img, H_img = xml_w, xml_h
            img_arr = np.zeros((H_img, W_img, 3), dtype=np.uint8)

        # XML-размеры «переставлены» относительно картинки?
        swapped = (xml_w > 0 and xml_h > 0 and xml_w == H_img and xml_h == W_img)

        def voc_xyxy_to_xywh_inclusive(xmin, ymin, xmax, ymax, W, H):
            # VOC обычно 1-based inclusive -> 0-based inclusive + xywh (+1 к ширине/высоте)
            x = max(0.0, float(xmin) - 1.0)
            y = max(0.0, float(ymin) - 1.0)
            w = max(0.0, float(xmax) - float(xmin) + 1.0)
            h = max(0.0, float(ymax) - float(ymin) + 1.0)
            if W > 0 and H > 0:
                x = max(0.0, min(x, W - 1.0))
                y = max(0.0, min(y, H - 1.0))
                w = max(0.0, min(w, W - x))
                h = max(0.0, min(h, H - y))
            return x, y, w, h

        def rotate90_ccw_xywh(x, y, w, h, Wsrc):
            # поворот прямоугольника на 90° CCW в 0-based inclusive системе
            x0, y0 = x, y
            x1, y1 = x + w - 1.0, y + h - 1.0
            pts = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)
            xr = pts[:, 1]
            yr = (Wsrc - 1.0) - pts[:, 0]
            xr0, yr0, xr1, yr1 = xr.min(), yr.min(), xr.max(), yr.max()
            nx, ny = xr0, yr0
            nw, nh = xr1 - xr0 + 1.0, yr1 - yr0 + 1.0
            return nx, ny, nw, nh

        anns: List[COCOAnnotation] = []
        cats: List[str] = []

        for obj in root.findall("object"):
            label = (obj.findtext("name") or "object").strip()
            cats.append(label)
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin", "0")) if bnd is not None else 0.0
            ymin = float(bnd.findtext("ymin", "0")) if bnd is not None else 0.0
            xmax = float(bnd.findtext("xmax", str(xml_w or W_img))) if bnd is not None else float(xml_w or W_img)
            ymax = float(bnd.findtext("ymax", str(xml_h or H_img))) if bnd is not None else float(xml_h or H_img)

            if swapped and xml_w > 0 and xml_h > 0:
                # 1) переводим VOC-бокс в систему XML (xml_w x xml_h)
                x, y, w, h = voc_xyxy_to_xywh_inclusive(xmin, ymin, xmax, ymax, W=xml_w, H=xml_h)
                # 2) поворачиваем в систему реального изображения (W_img x H_img)
                x, y, w, h = rotate90_ccw_xywh(x, y, w, h, Wsrc=xml_w)
                # 3) клиппим по реальным размерам
                x = max(0.0, min(x, W_img - 1.0))
                y = max(0.0, min(y, H_img - 1.0))
                w = max(0.0, min(w, W_img - x))
                h = max(0.0, min(h, H_img - y))
            else:
                # Обычный путь (без перестановки осей)
                x, y, w, h = voc_xyxy_to_xywh_inclusive(xmin, ymin, xmax, ymax, W=W_img, H=H_img)

            area = float(w * h)
            mask_np = np.zeros((H_img, W_img), dtype=np.uint8)
            x1, y1 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
            x2 = int(min(W_img, np.ceil(x + w)))
            y2 = int(min(H_img, np.ceil(y + h)))
            if x2 > x1 and y2 > y1:
                mask_np[y1:y2, x1:x2] = 1

            anns.append(COCOAnnotation(
                img=img_arr,
                mask=mask_np,
                label=label,
                image_size=(int(W_img), int(H_img)),
                width=int(W_img),
                height=int(H_img),
                area=area,
                file_name=file_name,
                bbox=[x, y, w, h],
            ))

        return anns, cats