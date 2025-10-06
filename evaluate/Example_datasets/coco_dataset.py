import sys
from pathlib import Path
from typing import Any, List, Dict
import numpy as np
from PIL import Image, ImageDraw
import json
from evaluate.dataset import Dataset
from evaluate.dataset_model import DatasetModel
from evaluate.dataset_meta import DatasetMeta
from evaluate.coco_annotation import CocoAnnotation
from evaluate.log_utils import get_logger
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

logger = get_logger(__name__)
        
class CocoDataset(Dataset):
    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "CocoDataset":
        annotations = cls.build_annotations(obj, base_dir=None)
        existing_meta = obj.get('meta', {})
        
        meta = DatasetMeta(
            dataset_type='coco_dataset',
            total_images=len(obj.get('images', [])),
            total_annotations=len(obj.get('annotations', [])),
            categories=obj.get('categories', []),
            uid=existing_meta.get('uid'),
            name=existing_meta.get('name'),
            url=existing_meta.get('url'),
            color_channels=existing_meta.get('color_channels', []),
            source_path=existing_meta.get('source_path'),
            base_directory=existing_meta.get('base_directory'),
            extra=existing_meta.get('extra', {})
        )
        
        data = DatasetModel(
            data_points=annotations,
            meta=meta
        )
        return cls(dataset=data)
    
    @classmethod
    def from_path(cls, path: Path, **kwargs: Any) -> "CocoDataset":
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        
        base_dir = Path(path).parent
        annotations = cls.build_annotations(obj, base_dir=base_dir)
        existing_meta = obj.get('meta', {})
        
        meta = DatasetMeta(
            dataset_type='chicken_detection',
            source_path=str(path),
            base_directory=str(base_dir),
            total_images=len(obj.get('images', [])),
            total_annotations=len(obj.get('annotations', [])),
            categories=obj.get('categories', []),
            uid=existing_meta.get('uid'),
            name=existing_meta.get('name'),
            url=existing_meta.get('url'),
            color_channels=existing_meta.get('color_channels', []),
            extra=existing_meta.get('extra', {})
        )
        
        data = DatasetModel(
            data_points=annotations,
            meta=meta
        )
        return cls(dataset=data)

    @classmethod
    def build_annotations(cls, obj: dict[str, Any], base_dir: Path | None) -> List[CocoAnnotation]:
        images_by_identifier: Dict[int, Dict[str, Any]] = {int(im.get('id')): im for im in obj.get('images', []) if 'id' in im}
        categories_by_identifier: Dict[int, Dict[str, Any]] = {int(cat.get('id')): cat for cat in obj.get('categories', []) if 'id' in cat}
        anns: List[CocoAnnotation] = []
        for ann in obj.get('annotations', []) or []:
            try:
                image_id = int(ann.get('image_id'))
                image_info = images_by_identifier.get(image_id)
                if not image_info:
                    continue
                file_name: str = image_info.get('file_name', '')
                width: int = int(image_info.get('width', 0) or 0)
                height: int = int(image_info.get('height', 0) or 0)
                cat_id = ann.get('category_id')
                cat = categories_by_identifier.get(int(cat_id)) if cat_id is not None else None
                label: int | str
                if isinstance(cat, dict):
                    name_value = cat.get('name')
                    if isinstance(name_value, str):
                        label = name_value
                    else:
                        label = int(cat_id) if cat_id is not None else -1
                else:
                    label = int(cat_id) if cat_id is not None else -1
                bounding_box_list = ann.get('bbox') or []
                bbox: List[float] = [float(v) for v in bounding_box_list] if isinstance(bounding_box_list, (list, tuple)) else []
                if len(bbox) == 4:
                    x, y, w, h = bbox
                else:
                    x = y = 0.0
                    w = float(width)
                    h = float(height)
                    bbox = [x, y, w, h]
                area_value = ann.get('area')
                if area_value is None:
                    area = float(w * h)
                else:
                    try:
                        area = float(area_value)
                    except Exception:
                        area = float(w * h)
                mask_np = np.zeros((height, width), dtype=np.uint8)
                segm = ann.get('segmentation')
                def draw_polygon_mask(points: List[float]) -> None:
                    if not points or len(points) < 6:
                        return
                    xy = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
                    pil_mask = Image.fromarray(mask_np, mode='L')
                    draw = ImageDraw.Draw(pil_mask)
                    draw.polygon(xy, outline=1, fill=1)
                    np.copyto(mask_np, np.array(pil_mask, dtype=np.uint8))
                if isinstance(segm, list) and len(segm) > 0:
                    for poly in segm:
                        if isinstance(poly, list):
                            draw_polygon_mask(poly)
                elif isinstance(segm, dict) and 'counts' in segm and 'size' in segm:
                    rle_h, rle_w = (int(segm['size'][0]), int(segm['size'][1]))
                    if rle_h == height and rle_w == width:
                        pass
                if mask_np.sum() == 0 and len(bbox) == 4:
                    x0, y0, bw, bh = bbox
                    x1, y1 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
                    x2 = int(min(width, np.ceil(x0 + bw)))
                    y2 = int(min(height, np.ceil(y0 + bh)))
                    if x2 > x1 and y2 > y1:
                        mask_np[y1:y2, x1:x2] = 1
                if base_dir is not None and file_name:
                    img_path = (base_dir / file_name)
                    try:
                        image_array = np.array(Image.open(img_path).convert('RGB'))
                    except Exception:
                        image_array = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    image_array = np.zeros((height, width, 3), dtype=np.uint8)
                anns.append(
                    CocoAnnotation(
                        img=image_array,
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
            except Exception as e:
                logger.warning(f"Не удалось обработать аннотацию: {e}. Аннотация: {ann}")
                continue

        return anns