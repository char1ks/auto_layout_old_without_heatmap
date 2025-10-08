from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.eval import Eval
from evaluation.log_utils import get_logger
from evaluation.example_datasets.voc_dataset import VocDataset

from flashbone.core.detection.searchdet_detector import SearchDetDetector
from ultralytics import FastSAM
from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz
from flashbone.core.segmentation import SamSegmenter, SegmenterConfig
from flashbone.core.heatmap_generation import HeatmapGenerator
from flashbone.core.classification.mask_classifier_knn import MaskClassifierKNN
from flashbone.core.image_resizing import ImageResizer, ResizeContext

logger = get_logger(__name__)


def build_detector() -> SearchDetDetector:
    encoder = DinoV3EncoderGaz()
    heatmap_generator = HeatmapGenerator(
        dino_fe=encoder,
        use_cosine_similarity_for_heatmap=True,
        threshold_cosine=0.3,
    )
    sam_model = FastSAM("FastSAM-x.pt")
    seg_cfg = SegmenterConfig(
        min_mask_area=200,
        confidence_threshold=0.5,
        iou_threshold=0.8,
        mask_threshold=0.5,
    )
    segmenter = SamSegmenter(sam_model=sam_model, config=seg_cfg)
    classifier = MaskClassifierKNN(encoder=encoder, d=1024)
    image_resizer = ImageResizer(max_side=1024)
    # ensure ResizeContext compatibility
    _orig_resize = image_resizer.resize
    image_resizer.resize = lambda img: (
        (lambda o, c: (o, ResizeContext(scale=float(c.get("scale", 1.0)), orig_shape=tuple(c.get("orig_shape", o.shape[:2])))) if isinstance(c, dict) else (o, c))
    )(*_orig_resize(img))
    return SearchDetDetector(
        segmenter=segmenter,
        classifier=classifier,
        heatmap_generator=heatmap_generator,
        image_resizer=image_resizer,
    )


def build_dataset(root: Optional[Path] = None) -> VocDataset:
    root = root or Path("fruits")
    ann_dir = root / "annotations"
    img_dir = root / "images"
    return VocDataset.from_path(root, ann_dir=ann_dir, img_dir=img_dir)


def main():
    dataset = build_dataset()
    detector = build_detector()
    eva = Eval(dataset=dataset, detector=detector, metrics=["mAP", "mIoU", "dice", "clf_report"])
    preds, metrics, out_dir = eva.run(
        positive_dir=Path("examples/positive"),
        negative_dir=None,
        image_root=dataset.root,
        average="micro",
        output_dir=Path("results_cache"),
        dump_report=True,
        enable_profile=False,
    )
    logger.info(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()