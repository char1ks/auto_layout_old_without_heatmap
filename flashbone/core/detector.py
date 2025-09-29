import numpy as np
from PIL import Image
import torch

# from flashbone.core.filtering import MaskFilter  
from flashbone.core.segmentation import SamSegmenter
from flashbone.core.classifier import ClassifierKNN, ClassData
from flashbone.core.heatmap_generation import HeatmapGenerator
from flashbone.core.detector_base import DetectorBase, DetectionResult


class SearchDetDetector(DetectorBase):
    def __init__(self, segmenter: SamSegmenter, classifier: ClassifierKNN, heatmap_generator: HeatmapGenerator) -> None:
        # self.mask_filter = MaskFilter(self.params) # TODO: add and test masks filter
        self._segmenter = segmenter
        self._classifier = classifier
        self._heatmap_generator = heatmap_generator

    def _bbox_from_mask(self, mask: np.ndarray) -> list[int]:
        seg = (mask > 0).astype(bool)
        ys, xs = np.where(seg)
        if xs.size and ys.size :
            x_min , x_max = int(xs.min()), int(xs.max())
            y_min , y_max = int(ys.min()), int(ys.max())
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        else :
            bbox = [0, 0, 0, 0]
        return bbox

    def set_references(self, pos_by_class: dict[str, list[Image.Image]], neg_imgs: list[Image.Image]) -> None:
        positives = []
        for cl, imgs in pos_by_class.items():
            self._classifier.index([
                ClassData(
                    class_id=int(cl), 
                    images=imgs, 
                    negative_examples=neg_imgs,
                ),
            ])
            positives.extend(imgs)
        self._heatmap_generator.init_pooled_features_train(
            positive_images=positives, negative_images=neg_imgs)

    def find_present_elements(self, image: Image.Image) -> list[DetectionResult]:
        _, heatmap_resized = self._heatmap_generator.generate_heatmap(image)
        heatmap_resized = self._heatmap_generator.apply_threshold(heatmap_resized, threshold_dotp=10) # TODO: (@gas) parametrize thresholds
        heatmap_np = heatmap_resized.cpu().numpy()
        masks = self._segmenter.segment(image, heatmap=heatmap_np)

        result = []
        for mask in masks:
            bbox = self._bbox_from_mask(mask)
            cls_preds = self._classifier.predict(
                images=[image],
                masks=[Image.fromarray(mask)],
                threshold=0.4,
                mask_threshold=0.5,
            )
            md = DetectionResult(
                mask=mask,
                bbox=bbox,
                area=mask.sum(),
                # NOTE: (@gas) [0] bc a batch of 1 used to process a single input image
                score=cls_preds[0].score, 
                class_id=cls_preds[0].class_id,
            )
            result.append(md)

        return result


if __name__=="__main__":
    # TODO: (@gas) convert to tests

    import time 
    from ultralytics import FastSAM

    from flashbone.core.encoding import DinoV3EncoderGaz
    from flashbone.core.segmentation import SegmenterConfig
    from flashbone.core.heatmap_generation import HeatmapGenerator, crop_by_mask
    from flashbone.core.classifier import ClassifierKNN

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
    sam_model = FastSAM('FastSAM-x.pt')
    encoder = DinoV3EncoderGaz()
    # warmup
    _ = encoder.encode([img_pil_ex])
    heatmap_generator = HeatmapGenerator(dino_fe=encoder, use_cosine_similarity_for_heatmap=False)
    sam = SamSegmenter(
        sam_model=sam_model,
        config=SegmenterConfig(
            min_mask_area=200,
            confidence_threshold=0.5,
            iou_threshold=0.8,
            mask_threshold=0.5,
        )
    )
    classifier = ClassifierKNN(encoder=encoder, d=1024)
    detector = SearchDetDetector(segmenter=sam, classifier=classifier, heatmap_generator=heatmap_generator)
    # ---

    detector.set_references(
        pos_by_class={0: [train_image_pos]}, neg_imgs=[train_image_neg_1, train_image_neg_2])

    start = time.perf_counter()
    results = detector.find_present_elements(img_pil_right)
    end = time.perf_counter()
    print(f"{int((end-start)*1000)} ms.") 

    print()
    print(img_pil_right.size, len(results))
    for res in results:
        print((res.class_id, res.score, res.bbox, res.area))
