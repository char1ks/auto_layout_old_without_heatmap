# NOTE: (@gas) this code is changed and refactored version of the searchdet: https://github.com/Mankeerat/SearchDet/blob/main/heatmap_generation.py
import logging

import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from einops import reduce
import torch.nn.functional as F

from flashbone.core.encoding import DinoV3EncoderGaz


_logger = logging.getLogger("heatmap-generation")


def calculate_attention_weights_softmax(query_embedding, example_embeddings):
    # Ensure query_embedding is a 2D array
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Ensure example_embeddings is a 2D array
    if example_embeddings.ndim == 1:
        example_embeddings = example_embeddings.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, example_embeddings).flatten()
    exp_similarities = np.exp(similarities)
    attention_weights = exp_similarities / np.sum(exp_similarities)
    return attention_weights


# Function to adjust the query embedding
def adjust_embedding(query_embedding, positive_embeddings, negative_embeddings):
    positive_weights = calculate_attention_weights_softmax(query_embedding, positive_embeddings)
    negative_weights = calculate_attention_weights_softmax(query_embedding, negative_embeddings)
    
    # Compute weighted sums of positive and negative embeddings
    positive_adjustment = np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
    negative_adjustment = np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)
    
    # Subtract negative adjustment from positive adjustment
    combined_adjustment = positive_adjustment - negative_adjustment
    
    return combined_adjustment


class HeatmapGenerator:
    def __init__(
        self,
        dino_fe: DinoV3EncoderGaz,
        attention_pool_examples: bool = False,
        use_cosine_similarity_for_heatmap: bool = True
    ):
        self.dino_fe = dino_fe
        self.attention_pool_examples = attention_pool_examples
        self.use_cosine_similarity_for_heatmap = use_cosine_similarity_for_heatmap
        self.pooled_patch_features_pos, self.pooled_patch_features_neg = None, None

    @torch.no_grad()
    def init_pooled_features_train(self, positive_images: list[Image.Image], negative_images: list[Image.Image] = []) -> None:
        self.pooled_patch_features_pos = self._generate_pooled_patch_features(positive_images) # (n, d)
        if negative_images:
            self.pooled_patch_features_neg = self._generate_pooled_patch_features(negative_images) # (n, d)

    @torch.no_grad()
    def generate_heatmap(self, input_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.dino_fe.encode([input_image])
        patch_feats = feats.patches[0].cpu()
        query_feats = reduce(patch_feats, 'd h w -> d', 'mean') # (d,)

        positive_embed = self._get_pooled_embed(query_feats, self.pooled_patch_features_pos).cpu()
        if self.pooled_patch_features_neg is None:
            negative_embed = torch.zeros_like(positive_embed).cpu()
        else:
            negative_embed = self._get_pooled_embed(query_feats, self.pooled_patch_features_neg).cpu()

        # Adjust the query embedding using positive and negative examples
        adjusted_embedding = adjust_embedding(
            query_feats.detach().cpu().numpy(), positive_embed.numpy(), negative_embed.numpy())
        adjusted_embedding = torch.tensor(adjusted_embedding, dtype=torch.float32)

        d = adjusted_embedding.shape[0]
        # Compute heatmap
        if self.use_cosine_similarity_for_heatmap:
            _logger.debug('Using cosine similarity for heatmap')
            heatmap = torch.cosine_similarity(patch_feats, adjusted_embedding.view(d, 1, 1), dim=0) # (d, h, w) X (d, 1, 1) --> (h, w)

        else: # Dot product; potentially more expressive, but requires more tuning of clamp_min, clamp_max, and scale
            _logger.debug('Using dot product for heatmap')
            heatmap = torch.einsum('cxy,c->xy', patch_feats, adjusted_embedding)

        img_w, img_h = input_image.size
        heatmap_resized = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0), 
            size=(img_h, img_w), 
            mode='bilinear',
        ).squeeze(0).squeeze(0)

        return heatmap, heatmap_resized

    def _get_pooled_embed(self, query_feats: torch.Tensor, pooled_patch_features: torch.Tensor):
        if self.attention_pool_examples:
            _logger.debug('Attention pooling example embeddings')
            return self._attention_pool_keys(query_feats, pooled_patch_features) # (d,)

        else: # Average pooling
            _logger.debug('Average pooling example embeddings')
            return reduce(pooled_patch_features, 'n d -> d', 'mean')

    def _generate_pooled_patch_features(self, images: list[Image.Image]):
        '''
            images: list[PIL.Image.Image] of length n of images to extract features from.

            Returns: (n, d) torch.Tensor of pooled patch features.
        '''
        feats = self.dino_fe.encode(images) # list[(b, d, h, w)]
        patch_feats_l = feats.patches
        patch_feats_l = [reduce(patch_feats, 'd h w -> d', 'mean') for patch_feats in patch_feats_l] # list[d]
        patch_feats = torch.stack(patch_feats_l) # (n, d)

        return patch_feats

    def _attention_pool_keys(self, query: torch.Tensor, keys: torch.Tensor):
        '''
            query: (d,)
            keys: (n, d)

            Returns: (d,) torch.Tensor of pooled keys based on attention weights between query and keys.
        '''
        weights = (keys @ query).softmax(dim=0) # (n,)
        weighted_keys = keys * weights.unsqueeze(1)
        pooled_keys = reduce(weighted_keys, 'n d -> d', 'sum')

        return pooled_keys


def crop_by_mask(image: Image.Image, mask: Image.Image):
    mask_array = np.array(mask)
    nonzero = np.nonzero(mask_array)
    if len(nonzero[0]) == 0:
        return image
    
    top = nonzero[0].min()
    bottom = nonzero[0].max() + 1
    left = nonzero[1].min()
    right = nonzero[1].max() + 1
    
    return image.crop((left, top, right, bottom))


def min_max_scale(array: np.ndarray) -> np.ndarray:
    min_vals = np.min(array, axis=0)
    max_vals = np.max(array, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Handle case where max = min
    scaled_array = (array - min_vals) / range_vals
    return scaled_array


if __name__=="__main__":
    import time 
    import cv2

    model = DinoV3EncoderGaz()
    heatmap_generator = HeatmapGenerator(dino_fe=model, use_cosine_similarity_for_heatmap=False)
 
    img_pil_ex = Image.open(".local/example.jpg").convert("RGB")
    # warmup
    feats_ex = model.encode([img_pil_ex])

    img_pil_left = Image.open(".local/image_left.jpg").convert("RGB")
    mask_left = Image.open(".local/image_left_fg.png")
    mask_left = mask_left.split()[-1]
    img_pil_right = Image.open(".local/image_right.jpg").convert("RGB")

    train_image_pos = crop_by_mask(img_pil_left, mask_left)
    # sky crop
    train_image_neg_1 = img_pil_left.crop((0, 0, 150, 150)) 
    # grass crop
    train_image_neg_2 = img_pil_left.crop((img_pil_left.width-150, img_pil_left.height-150, img_pil_left.width, img_pil_left.height))
    heatmap_generator.init_pooled_features_train(
        positive_images=[train_image_pos], negative_images=[train_image_neg_1, train_image_neg_2])

    start = time.perf_counter()
    heatmap, heatmap_resized = heatmap_generator.generate_heatmap(img_pil_right)
    # heatmap, heatmap_resized = heatmap_generator.generate_heatmap(img_pil_ex)
    end = time.perf_counter()
    print(heatmap_resized.shape, heatmap_resized.min(), heatmap_resized.max())
    print(f"{int((end-start)*1000)} ms.") 

    heatmap_np = heatmap_resized.numpy()

    # NOTE: (@gas) only for dot-product
    heatmap_np[heatmap_np < 5] = 0 # some initial thresholding
    heatmap_np /= np.abs(heatmap_np).max()

    # thresholding 
    heatmap_np[heatmap_np < 0.5] = 0

    cv2.imwrite(".local/crop_debug_gaz.png", cv2.cvtColor(np.asarray(train_image_pos), cv2.COLOR_RGB2BGR))
    cv2.imwrite(".local/heatmap_debug_gaz.png", (heatmap_np*255).astype(np.uint8))
