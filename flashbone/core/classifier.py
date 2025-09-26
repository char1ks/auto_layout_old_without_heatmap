from typing import Any
from dataclasses import dataclass, field

from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import faiss

from flashbone.core.encoding import DinoV3EncoderGaz


@dataclass
class ClassData:
    class_id: int
    images: list[Image.Image]
    negative_examples: list[Image.Image] = field(default_factory=list)


@dataclass
class ClassifierPrediction:
    class_id: int
    score: float


class ClassifierKNN:
    def __init__(self, encoder: DinoV3EncoderGaz, d: int = 1024) -> None:
        self._encoder = encoder
        self._d = d
        self._index: dict[str, faiss.IndexFlatIP] = dict()

    def _calculate_attention_weights_softmax(self, query_embedding, example_embeddings):
        similarities = cosine_similarity(query_embedding.reshape(1, -1), example_embeddings).flatten()
        exp_similarities = np.exp(similarities)
        attention_weights = exp_similarities / np.sum(exp_similarities)
        return attention_weights
    
    def _adjust_embedding(self, query_embedding, positive_embeddings, negative_embeddings):
        positive_weights = self._calculate_attention_weights_softmax(query_embedding, positive_embeddings)
        negative_weights = self._calculate_attention_weights_softmax(query_embedding, negative_embeddings)
    
        # Compute weighted sums of positive and negative embeddings
        positive_adjustment = np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
        negative_adjustment = np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)
    
        # Subtract negative adjustment from positive adjustment
        combined_adjustment = positive_adjustment - negative_adjustment
        return combined_adjustment

    def _index_single(self, class_id: int, query_imgs: list[Image.Image], negative_imgs: list[Image.Image] = []) -> None:
        positive_embeddings = [self._encoder.encode([img]).cls.cpu().numpy() for img in query_imgs]
        positive_embeddings = np.concatenate(positive_embeddings, axis=0)
    
        negative_embeddings = np.zeros((1, self._d), dtype=np.float32)
        if negative_imgs:
            negative_embeddings = [self._encoder.encode([img]).cls.cpu().numpy() for img in negative_imgs]
            negative_embeddings = np.concatenate(negative_embeddings, axis=0)
    
        # Adjust the query embedding for each query image
        adjusted_query_vectors = np.array([
            self._adjust_embedding(embedding, positive_embeddings, negative_embeddings)
            for embedding in positive_embeddings
        ])
    
        # Normalize query and mask vectors
        adjusted_query_vectors = adjusted_query_vectors / np.linalg.norm(adjusted_query_vectors, axis=1, keepdims=True)
    
        # Create FAISS index and add query vectors
        index = faiss.IndexFlatIP(self._d)  # Using inner product for cosine similarity
        index.add(adjusted_query_vectors)

        self._index[class_id] = index

    def index(self, dataset: list[ClassData]) -> None:
        for cls in dataset:
            self._index_single(cls.class_id, cls.images)

    def predict(self, images: list[Image.Image], masks: list[Image.Image] = [], threshold: float = 0.474, mask_threshold: float = 0.5, topk: int = 1) -> Any:
        # TODO: (@gas) adopt for batched encoding (`encoder` TODOs must be resolved before that)
        mask_vectors = [
            self._encoder.encode_mask(
                masks=[mask], images=[img], mask_threshold=mask_threshold).cpu().numpy() 
            for img, mask in zip(images, masks)
        ]
        mask_vectors = np.concatenate(mask_vectors, axis=0)
        mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)
        preds: list[ClassifierPrediction] = []
        for class_id, index in self._index.items():
            # Search for matches in the FAISS index
            similarities, indices = index.search(mask_vectors, topk)
            # Map similarities to [0, 1]
            normalized_similarities = np.squeeze((similarities + 1) / 2, 0)
            # Apply a threshold to filter matches
            filtered_indices = np.where(normalized_similarities > threshold)[0]
            if not len(filtered_indices):
                pred = ClassifierPrediction(
                    class_id=-1,
                    score=0,
                )
            else:
                pred = ClassifierPrediction(
                    class_id=class_id,
                    score=normalized_similarities[filtered_indices][0],
                )
            preds.append(pred)
        return preds


if __name__=="__main__":
    # TODO: (@gas) convert to tests

    img_pil_right = Image.open(".local/image_right.jpg").convert("RGB")
    # sky crop
    train_image_neg_1 = img_pil_right.crop((0, 0, 150, 150)) 
    # grass crop
    train_image_neg_2 = img_pil_right.crop(
        (img_pil_right.width-150, img_pil_right.height-150, img_pil_right.width, img_pil_right.height))

    encoder = DinoV3EncoderGaz()
    classifier = ClassifierKNN(encoder=encoder, d=1024)

    classifier.index([
        ClassData(
            class_id=0, 
            images=[img_pil_right], 
            negative_examples=[train_image_neg_1, train_image_neg_2],
        ),
    ])

    img_pil_left = Image.open(".local/image_left.jpg").convert("RGB")
    mask_left = Image.open(".local/image_left_fg.png")
    mask_left = mask_left.split()[-1]

    preds = classifier.predict(
        images=[img_pil_left],
        masks=[mask_left],
        threshold=0.4,
        mask_threshold=0.5,
    )
    print(">>> PREDS: ", preds)
