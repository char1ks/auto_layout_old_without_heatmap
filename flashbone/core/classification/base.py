from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import faiss
import numpy as np
from PIL import Image


@dataclass
class ClassData:
    class_id: int
    images: list[Image.Image] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    negative_images: list[Image.Image] = field(default_factory=list)
    negative_texts: list[Image.Image] = field(default_factory=list)


@dataclass
class ClassifierPredictRequest:
    images: list[Image.Image] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    threshold: float = 0.474
    mask_threshold: float = 0.3
    topk: int = 1


@dataclass
class ClassifierPrediction:
    class_id: int
    score: float


class ClassifierBase(ABC):
    def __init__(self, d: int, *args, **kwargs) -> None:
        self._d = d
        self._index: dict[str, faiss.IndexFlatIP] = dict()

    def _add_to_index(self, class_id: int, vecs: np.ndarray) -> None:
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(self._d)  # NOTE: (@gas) Using inner product for cosine similarity
        index.add(vecs)
        self._index[class_id] = index

    def _search(self, vecs: np.ndarray, threshold: float, topk: int) -> list[ClassifierPrediction]:
        preds: list[ClassifierPrediction] = []
        for class_id, index in self._index.items():
            # Search for matches in the FAISS index
            similarities, indices = index.search(vecs, topk)
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

    def index(self, dataset: list[ClassData]) -> None:
        for cls in dataset:
            vecs = self.get_adjusted_embeddings(cls)
            self._add_to_index(cls.class_id, vecs)

    def predict(self, req: ClassifierPredictRequest) -> list[ClassifierPrediction]:
        # TODO: (@gas) adopt for batched encoding (`encoder` TODOs must be resolved before that)
        vecs = self.encode(req)
        preds = self._search(vecs, req.threshold, req.topk) 
        return preds

    @abstractmethod
    def encode(self, req: ClassifierPredictRequest) -> np.ndarray:
        pass

    @abstractmethod
    def get_adjusted_embeddings(self, cls: ClassData) -> np.ndarray:
        pass
