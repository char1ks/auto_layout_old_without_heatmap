import cv2
import numpy as np
from PIL import Image

from flashbone.core.segmentation import SamSegmenter
from flashbone.core.classification.base import ClassData
from flashbone.core.classification.mask_classifier_knn import MaskClassifierKNN
from flashbone.core.classification.base import ClassifierPredictRequest
from flashbone.core.heatmap_generation import HeatmapGenerator
from flashbone.core.image_resizing import ImageResizer
from flashbone.core.detection.base import DetectorBase, DetectionResult
from evaluation.context import Context


class SearchDetDetector(DetectorBase):
    def __init__(
        self, 
        segmenter: SamSegmenter, 
        classifier: MaskClassifierKNN, 
        heatmap_generator: HeatmapGenerator,
        image_resizer: ImageResizer,
    ) -> None:
        self._segmenter = segmenter
        self._classifier = classifier
        self._heatmap_generator = heatmap_generator
        self._image_resizer = image_resizer

    def _bbox_from_mask(self, mask: np.ndarray) -> list[int]:
        seg = (mask > 0).astype(bool)
        ys, xs = np.where(seg)
        if xs.size and ys.size:
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        else:
            bbox = [0, 0, 0, 0]
        return bbox

    def _mask_to_polygons(self, mask: np.ndarray, min_area: int = 5) -> list[list[list[int]]]:
        """
        Convert a 2D binary mask (list-of-lists or ndarray) to polygons using cv2.findContours.
        Returns: List[List[[x, y], ...]] (one polygon = list of [x, y] points).
        Coordinates are in the mask grid space (0..W-1, 0..H-1).
    
        min_area filters tiny specks; tune as needed.
        """
        # OpenCV expects 0/255 for binary; ensure it:
        m = (mask > 0).astype(np.uint8) * 255
    
        # Find external contours; you can switch to RETR_TREE if you want holes/hierarchies
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < float(min_area):
                continue
            # Optional simplification (tune epsilon):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.001 * peri, True)
            pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
            if pts:
                polys.append(pts)
        return polys

    def set_references(self, pos_by_class: dict[str, list[Image.Image]], neg_imgs: list[Image.Image]) -> None:
        positives = []
        for cl, imgs in pos_by_class.items():
            self._classifier.index([
                ClassData(
                    class_id=int(cl), 
                    images=imgs, 
                    negative_images=neg_imgs,
                ),
            ])
            positives.extend(imgs)
        self._heatmap_generator.init_pooled_features_train(
            positive_images=positives, negative_images=neg_imgs)

    def detect(self, image: Image.Image, heatmap_threshold: float | None = None, class_threshold: float = 0.5, ctx: Context | None = None, *args, **kwargs) -> list[DetectionResult]:
        callback = kwargs.get("callback")
        span = (lambda name, fn: ctx.span(name, fn)) if ctx is not None else (lambda name, fn: fn())
        resized_img, resize_ctx = span("resize", lambda: self._image_resizer.resize(np.array(image)))
        resized_img_pil = Image.fromarray(resized_img)
        _, heatmap_resized = span("heatmap", lambda: self._heatmap_generator.generate_heatmap(resized_img_pil))

        heatmap_resized = span("threshold", lambda: self._heatmap_generator.apply_threshold(heatmap_resized, heatmap_threshold))
        heatmap_np = heatmap_resized.cpu().numpy()
        masks = span("segment", lambda: self._segmenter.segment(resized_img_pil, heatmap=heatmap_np))

        dets = []
        for mask in masks:
            bbox = self._bbox_from_mask(mask)
            cls_preds = span("classify", lambda: self._classifier.predict(
                ClassifierPredictRequest(
                    images=[resized_img_pil],
                    masks=[Image.fromarray(mask)],
                    threshold=class_threshold,
                )
            ))
            if len(cls_preds):
                # NOTE: (@gas) [0] bc a batch of 1 used to process a single input image
                cls_pred = cls_preds[0]
                if cls_pred.class_id < 0: # NOTE: (@gas) skip no class results
                    continue
                det = DetectionResult(
                    bbox=bbox,
                    area=int(mask.sum()),
                    polygons=self._mask_to_polygons(mask),
                    score=float(cls_pred.score), 
                    class_id=int(cls_pred.class_id),
                )
                dets.append(det)
        
        restored_dets = span("restore_dets", lambda: self._image_resizer.restore_dets(dets, resize_ctx))
        if ctx is not None:
            ctx.metrics = {
                "num_masks": int(len(masks)),
                "num_dets": int(len(restored_dets)),
            }
            ctx.finish(success=True)
            if callable(callback):
                callback(ctx)
        return restored_dets


if __name__=="__main__":
    # TODO: (@gas) convert to tests

    import time 
    from ultralytics import FastSAM

    from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz
    from flashbone.core.segmentation import SegmenterConfig
    from flashbone.core.heatmap_generation import HeatmapGenerator, crop_by_mask
    from flashbone.core.classification.mask_classifier_knn import MaskClassifierKNN
    from flashbone.core.image_resizing import ImageResizer

    img_pil_ex = Image.open(".local/example.jpg").convert("RGB")

    img_pil_left = Image.open(".local/image_left.jpg").convert("RGB")
    mask_left = Image.open(".local/image_left_fg.png")
    mask_left = mask_left.split()[-1]
    img_pil_right = Image.open(".local/image_right.jpg").convert("RGB")

    train_image_pos = crop_by_mask(img_pil_left, mask_left)
    # sky crop
    train_image_neg_1 = img_pil_left.crop((0, 0, 150, 150)) 
    # grass crop
    train_image_neg_2 = img_pil_left.crop((img_pil_left.width-150, img_pil_left.height-150, img_pil_left.width, img_pil_left.height))

    # ---
    # NOTE: (@gas) models instantiation - they're just shared across other classes
    sam_model = FastSAM('FastSAM-x.pt')
    encoder = DinoV3EncoderGaz()
    # warmup
    _ = encoder.encode([img_pil_ex])
    # ---

    # ---
    heatmap_generator = HeatmapGenerator(
        dino_fe=encoder, 
        # use_cosine_similarity_for_heatmap=False,
        # threshold_dotp=10, 
        use_cosine_similarity_for_heatmap=True,
        threshold_cosine=0.3, 
    )
    sam = SamSegmenter(
        sam_model=sam_model,
        config=SegmenterConfig(
            min_mask_area=200,
            confidence_threshold=0.5,
            iou_threshold=0.8,
            mask_threshold=0.5,
        )
    )
    classifier = MaskClassifierKNN(encoder=encoder, d=1024)
    image_resizer = ImageResizer(max_side=1024)
    detector = SearchDetDetector(
        segmenter=sam, 
        classifier=classifier, 
        heatmap_generator=heatmap_generator,
        image_resizer=image_resizer,
    )
    # ---

    detector.set_references(
        pos_by_class={0: [train_image_pos]}, neg_imgs=[train_image_neg_1, train_image_neg_2])

    start = time.perf_counter()
    from evaluation.context import Context
    ctx = Context(detector_name=detector.__class__.__name__, image_shape=tuple(np.array(img_pil_right).shape))
    results = detector.detect(
        img_pil_right, heatmap_threshold=0.3, class_threshold=0.4, ctx=ctx)
    end = time.perf_counter()
    print(f"{int((end-start)*1000)} ms.") 

    print()
    print(img_pil_right.size, len(results))
    for res in results:
        print((res.class_id, res.score, res.bbox, res.area))
