import numpy as np
from PIL import Image
import torch

# from flashbone.core.filtering import MaskFilter  
from flashbone.core.segmentation import SamSegmenter
from flashbone.core.classifier import ClassifierKNN, ClassData
from flashbone.core.heatmap_generation import HeatmapGenerator
from flashbone.detector_base import DetectorBase, DetectionResult


class SearchDetDetector(DetectorBase):
    def __init__(self, segmenter: SamSegmenter, classifier: ClassifierKNN, heatmap_generator: HeatmapGenerator) -> None:
        # self.mask_filter = MaskFilter(self.params) # TODO: add and test masks filter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def find_present_elements(self, image_np: np.ndarray) -> list[DetectionResult]:
        image_pil = Image.fromarray(image_np)

        _, heatmap_resized = self._heatmap_generator.generate_heatmap(image_pil)
        heatmap_resized = self._heatmap_generator.apply_threshold(heatmap_resized, threshold_dotp=10) # TODO: (@gas) parametrize thresholds
        heatmap_np = heatmap_resized.cpu().numpy()
        masks = self._segmenter.segment(image_pil, heatmap=heatmap_np)

        result = []
        for mask in masks:
            bbox = self._bbox_from_mask(mask)
            cls_preds = self._classifier.predict(
                images=[image_pil],
                masks=[mask],
                threshold=0.4,
                mask_threshold=0.5,
            )
            md = DetectionResult(
                mask=mask,
                bbox=bbox,
                area=mask.sum(),
                # NOTE: (@gas) [0] bc a batch of 1 used to classify
                score=cls_preds[0].score, 
                class_id=cls_preds[0].class_id,
            )
            result.append(md)

        return result


if __name__=="__main__":
    # TODO: (@gas) test
    pass
