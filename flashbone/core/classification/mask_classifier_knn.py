from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz
from flashbone.core.classification.base import ClassifierBase, ClassData, ClassifierPrediction, ClassifierPredictRequest


class MaskClassifierKNN(ClassifierBase):
    def __init__(self, encoder: DinoV3EncoderGaz, d: int = 1024, *args, **kwargs) -> None:
        self._encoder = encoder
        super().__init__(d=d, *args, **kwargs)

    def _calculate_attention_weights_softmax(self, query_embedding: np.ndarray, example_embeddings: np.ndarray) -> np.ndarray:
        similarities = cosine_similarity(query_embedding.reshape(1, -1), example_embeddings).flatten()
        exp_similarities = np.exp(similarities)
        attention_weights = exp_similarities / np.sum(exp_similarities)
        return attention_weights
    
    def _adjust_embedding(self, query_embedding: np.ndarray, positive_embeddings: np.ndarray, negative_embeddings: np.ndarray) -> np.ndarray:
        positive_weights = self._calculate_attention_weights_softmax(query_embedding, positive_embeddings)
        negative_weights = self._calculate_attention_weights_softmax(query_embedding, negative_embeddings)
    
        # Compute weighted sums of positive and negative embeddings
        positive_adjustment = np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
        negative_adjustment = np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)
    
        # Subtract negative adjustment from positive adjustment
        combined_adjustment = positive_adjustment - negative_adjustment
        return combined_adjustment

    def encode(self, req: ClassifierPredictRequest) -> list[ClassifierPrediction]:
        # TODO: (@gas) adopt for batched encoding (`encoder` TODOs must be resolved before that)
        if len(req.masks):
            vecs = [
                self._encoder.encode_mask(
                    masks=[mask], images=[img], mask_threshold=req.mask_threshold).cpu().numpy() 
                for img, mask in zip(req.images, req.masks)
            ]
        else:
            vecs = [
                self._encoder.encode(images=[img]).cls.cpu().numpy()
                for img in req.images
            ]
        vecs = np.concatenate(vecs, axis=0)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def get_adjusted_embeddings(self, cls: ClassData) -> np.ndarray:
        positive_embeddings = [self._encoder.encode([img]).cls.cpu().numpy() for img in cls.images] # NOTE: (@gas) query images
        positive_embeddings = np.concatenate(positive_embeddings, axis=0)
    
        negative_embeddings = np.zeros((1, self._d), dtype=np.float32)
        if cls.negative_images:
            negative_embeddings = [self._encoder.encode([img]).cls.cpu().numpy() for img in cls.negative_images]
            negative_embeddings = np.concatenate(negative_embeddings, axis=0)
    
        # Adjust the query embedding for each query image
        adjusted_query_vectors = np.array([
            self._adjust_embedding(embedding, positive_embeddings, negative_embeddings)
            for embedding in positive_embeddings
        ])
        return adjusted_query_vectors


if __name__=="__main__":
    # TODO: (@gas) convert to tests

    img_pil_right = Image.open(".local/image_right.jpg").convert("RGB")
    # sky crop
    train_image_neg_1 = img_pil_right.crop((0, 0, 150, 150)) 
    # grass crop
    train_image_neg_2 = img_pil_right.crop(
        (img_pil_right.width-150, img_pil_right.height-150, img_pil_right.width, img_pil_right.height))

    encoder = DinoV3EncoderGaz()
    classifier = MaskClassifierKNN(encoder=encoder, d=1024)

    classifier.index([
        ClassData(
            class_id=0, 
            images=[img_pil_right], 
            negative_images=[train_image_neg_1, train_image_neg_2],
        ),
    ])

    img_pil_left = Image.open(".local/image_left.jpg").convert("RGB")
    mask_left = Image.open(".local/image_left_fg.png")
    mask_left = mask_left.split()[-1]

    preds = classifier.predict(
        ClassifierPredictRequest(
            images=[img_pil_left],
            masks=[mask_left],
            threshold=0.4,
            mask_threshold=0.5,
        )
    )
    print(">>> PREDS: ", preds)
