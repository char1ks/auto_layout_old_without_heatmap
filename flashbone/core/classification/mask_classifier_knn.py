from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz
from flashbone.core.classification.base import ClassifierBase, ClassData, ClassifierPredictRequest


class MaskClassifierKNN(ClassifierBase):
    def __init__(self, encoder: DinoV3EncoderGaz, d: int = 1024, *args, **kwargs) -> None:
        self._encoder = encoder
        super().__init__(d=d, *args, **kwargs)

    def _calculate_attention_weights_softmax(
        self,
        queries: np.ndarray,             # (Q, D)
        example_embeddings: np.ndarray,  # (E, D)
    ) -> np.ndarray:                     # (Q, E)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        sims = cosine_similarity(queries, example_embeddings)  
        
        sims = sims - sims.max(axis=1, keepdims=True)
        exps = np.exp(sims)
        weights = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)
        return weights

    
    def _adjust_embedding(
        self,
        query_embeddings: np.ndarray,     # (Q, D) 
        positive_embeddings: np.ndarray,  # (P, D)
        negative_embeddings: np.ndarray,  # (N, D)
    ) -> np.ndarray:                      # (Q, D) 
        single = False
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            single = True

        w_pos = self._calculate_attention_weights_softmax(query_embeddings, positive_embeddings)  # (Q, P)
        positive_adjustment = w_pos @ positive_embeddings                                         # (Q, D)

        if negative_embeddings is not None and negative_embeddings.size > 0:
            w_neg = self._calculate_attention_weights_softmax(query_embeddings, negative_embeddings)  # (Q, N)
            negative_adjustment = w_neg @ negative_embeddings                                         # (Q, D)
        else:
            negative_adjustment = np.zeros_like(positive_adjustment)                                  # (Q, D)

        combined_adjustment = positive_adjustment - negative_adjustment  # (Q, D)
        return combined_adjustment[0] if single else combined_adjustment



    #NOTE: (aod) Batching add
    def encode(self, req: ClassifierPredictRequest) -> np.ndarray:
        if len(req.masks):
            vecs_t = self._encoder.encode_mask(
                masks=req.masks,
                images=req.images,
                mask_threshold=req.mask_threshold
            )  # (B, D) torch
        else:
            vecs_t = self._encoder.encode(req.images).cls  # (B, D) torch

        vecs = vecs_t.detach().cpu().numpy()
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs  # (B, D)


    def get_adjusted_embeddings(self, cls: ClassData) -> np.ndarray:
        pos_t = self._encoder.encode(cls.images).cls          # torch (P, D)
        positive_embeddings = pos_t.detach().cpu().numpy()    #(P, D)
        if cls.negative_images:
            neg_t = self._encoder.encode(cls.negative_images).cls   # torch (N, D)
            negative_embeddings = neg_t.detach().cpu().numpy()      #(N, D)
        else:
            negative_embeddings = np.empty((0, self._d), dtype=np.float32)
        adjusted_query_vectors = self._adjust_embedding(
            positive_embeddings,          # queries: (P, D)
            positive_embeddings,          # positives: (P, D)
            negative_embeddings,          # negatives: (N, D) или (0, D)
        )  # (P, D)
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
