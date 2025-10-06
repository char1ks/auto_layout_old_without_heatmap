from PIL import Image
import numpy as np

from flashbone.core.encoding.dinov3.multimodal import DinoV3VisionTextEncoderGaz
from flashbone.core.classification.base import ClassifierBase, ClassData, ClassifierPrediction, ClassifierPredictRequest


class VisionTextClassifierKNN(ClassifierBase):
    def __init__(self, encoder: DinoV3VisionTextEncoderGaz, d: int = 1024, *args, **kwargs) -> None:
        self._encoder = encoder
        super().__init__(d=d, *args, **kwargs)

    def _get_concat_embeddings(self, images: list[Image.Image], texts: list[str]) -> np.ndarray:
        vecs = vecs_images = vecs_texts = np.zeros((1, self._d), dtype=np.float32)
        if len(images):
            vecs_images = self._encoder.encode_images(images).cls.cpu().numpy()
            vecs = vecs_images
        if len(texts):
            vecs_texts = self._encoder.encode_texts(texts).cpu().numpy()
            vecs = vecs_texts
        if len(texts) and len(images):
            vecs = np.concatenate([vecs_images, vecs_texts], axis=0)
        return vecs

    def encode(self, req: ClassifierPredictRequest) -> list[ClassifierPrediction]:
        vecs = self._get_concat_embeddings(req.images, req.texts)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def get_adjusted_embeddings(self, cls: ClassData) -> np.ndarray:
        positive_embeddings = self._get_concat_embeddings(cls.images, cls.texts)
        negative_embeddings = self._get_concat_embeddings(cls.negative_images, cls.negative_texts)
        adjusted_query_vectors = (positive_embeddings.mean(axis=0) - negative_embeddings.mean(axis=0))[np.newaxis, ...]
        return adjusted_query_vectors


if __name__=="__main__":
    # TODO: (@gas) convert to tests

    img_pil_right = Image.open(".local/image_right.jpg").convert("RGB")
    # sky crop
    train_image_neg_1 = img_pil_right.crop((0, 0, 150, 150)) 
    # grass crop
    train_image_neg_2 = img_pil_right.crop(
        (img_pil_right.width-150, img_pil_right.height-150, img_pil_right.width, img_pil_right.height))

    encoder = DinoV3VisionTextEncoderGaz()
    classifier = VisionTextClassifierKNN(encoder=encoder, d=2048)

    # # NOTE: (@gas) images only (equivalent to MaskClassifierKNN)
    # classifier.index([
    #     ClassData(
    #         class_id=0, 
    #         images=[img_pil_right], 
    #         negative_images=[train_image_neg_1, train_image_neg_2],
    #     ),
    # ])
    # # NOTE: (@gas) multimodal neg
    # classifier.index([
    #     ClassData(
    #         class_id=0, 
    #         images=[img_pil_right], 
    #         # negative_images=[train_image_neg_1, train_image_neg_2],
    #         negative_images=[train_image_neg_1],
    #         negative_texts=["blue sky"],
    #     ),
    # ])
    # # NOTE: (@gas) multimodal pos and neg
    # classifier.index([
    #     ClassData(
    #         class_id=0, 
    #         texts=["donkey"],
    #         negative_images=[train_image_neg_1],
    #         negative_texts=["blue sky"],
    #     ),
    # ])
    # NOTE: (@gas) text only
    classifier.index([
        ClassData(
            class_id=0, 
            texts=["donkey"],
            negative_texts=["green grass", "blue sky"],
        ),
    ])
    # NOTE: (@gas) any query should return class 0 with score >=0.4

    img_pil_left = Image.open(".local/image_left.jpg").convert("RGB")

    print(" --- Inference --- ")
    preds = classifier.predict(
        ClassifierPredictRequest(
            images=[img_pil_left],
            threshold=0.3,
        )
    )
    print(">>> PREDS: ", preds)
