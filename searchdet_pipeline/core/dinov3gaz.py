from dataclasses import dataclass
import sys
from pathlib import Path
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Any ,Dict ,Optional
import re
import math
from enum import Enum

try :
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception :
    pass

import torchvision.transforms.functional as TF

# try:
#     from dinov3.hub import backbones as dino_backbones
# except ImportError:
#     project_root = Path(__file__ ).resolve().parent.parent.parent
#     dinov3_repo_path = project_root
#     if str(dinov3_repo_path) not in sys.path:
#         sys.path.insert(0, str(dinov3_repo_path))
#     inner_dinov3_path = project_root / 'dinov3'
#     if str(inner_dinov3_path) not in sys.path :
#         sys.path.insert(0, str(inner_dinov3_path))
#     from dinov3.hub import backbones as dino_backbones
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.data.transforms import make_classification_eval_transform

PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"


MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

# TODO: (@gas) automate
DINOV3_LOCATION = "/home/synetra/ml_segmentation/vendor/dinov3"


class DinoV3VisionTextEncoderGaz:
    def __init__(self):
        self.model, self.tokenizer = \
            dinov3_vitl16_dinotxt_tet1280d20h24l(bpe_path_or_url="tokenizers/bpe_simple_vocab_16e6.txt.gz")
        model = model.cuda()
        self.preprocessor = make_classification_eval_transform()
 
    def encode(self, img: Image.Image, texts: list[str] = []) -> tuple[torch.Tensor, ...]:
        image_tensor = torch.stack([self.preprocessor(img)], dim=0).cuda()
        if texts:
            tokenized_texts_tensor = self.tokenizer.tokenize(texts).cuda() 
            with torch.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    image_features, patch_tokens, bb_patch_tokens = model.encode_image_with_patch_tokens(image_tensor)
                    text_features = model.encode_text(tokenized_texts_tensor)
            return image_features, patch_tokens, bb_patch_tokens, text_features
        else:
            with torch.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    image_features, patch_tokens, bb_patch_tokens = model.encode_image_with_patch_tokens(image_tensor)
            return image_features, patch_tokens, bb_patch_tokens


def resize_transform(
    mask_image: Image.Image,
    image_size: int = 768,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """
    image resize transform to dimensions divisible by patch size
    """
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


@dataclass
class DinoFeaturesPT:
    cls: torch.Tensor
    patches: torch.Tensor


class DinoV3EncoderGaz:
    def __init__(self, model_name: str = MODEL_DINOV3_VITL):
        self.n_layers = MODEL_TO_NUM_LAYERS[model_name]
        self.model = torch.hub.load(
            repo_or_dir=DINOV3_LOCATION,
            model=model_name,
            source="local",
        )
        self.model.eval()
        self.model.cuda()
        # NOTE: (@gas) for masks only
        self.patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
        self.patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

    def encode(self, img_pil: Image.Image) -> DinoFeaturesPT:
        """
        rgb image --> tuple(patch features, cls feature vector)
        shapes:
            (H, W, CH) --> tuple((B, D, H, W), (B, D)) --> __tuple((D, H, W), (,D))__
        """
        image_resized = resize_transform(img_pil)
        image_resized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        image_resized = image_resized.unsqueeze(0).cuda()
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = self.model.get_intermediate_layers(
                    image_resized, 
                    n=range(self.n_layers), 
                    return_class_token=True,
                    reshape=True, 
                    norm=True,
                )[-1]
        return DinoFeaturesPT(cls=feats[1].squeeze(), patches=feats[0].squeeze())

    def encode_mask(
        self, 
        mask: Image.Image, 
        features: DinoFeaturesPT | None = None, 
        img_pil: Image.Image | None = None, 
        mask_threshold: float = 0.5,
    ) -> torch.Tensor:
        if features is None and img_pil is None:
            raise ValueError("either of features or img_pil should be passed, got none of them")
        mask_resized = resize_transform(mask)
        mask_quantized = self.patch_quant_filter(mask_resized.unsqueeze(0)).squeeze().detach().cpu()
        if features is None and img_pil is not None:
            features = self.encode(img_pil)
        patches_fg_selection = (mask_quantized > mask_threshold) # (P_H, P_W); bool
        patches_selected = features.patches[:, patches_fg_selection] # (D, H, W) --> (D, N_points)
        patches_mean = patches_selected.mean(axis=1) # (,D)
        return patches_mean


def cosine_similarity_pt(A, B):
    if A.shape[-1] != B.shape[-1]:
        raise ValueError("Last dimension of A and B must match (feature dimension d)")

    if A.dim() == 1 and B.dim() == 1:
        A_norm = A / (torch.norm(A, dim=-1, keepdim=True) + 1e-8)
        B_norm = B / (torch.norm(B, dim=-1, keepdim=True) + 1e-8)
        return torch.dot(A_norm, B_norm)
    
    if A.dim() < 2 or B.dim() < 2:
        raise ValueError("For batched inputs, tensors must have at least 2 dimensions (..., n, d)")

    A_norm = A / (torch.norm(A, dim=-1, keepdim=True) + 1e-8)
    B_norm = B / (torch.norm(B, dim=-1, keepdim=True) + 1e-8)
    
    similarity = torch.matmul(A_norm, B_norm.mT)
    return similarity


if __name__=="__main__":
    import time 

    model = DinoV3EncoderGaz()
 
    img_pil_ex = Image.open(".local/example.jpg").convert("RGB")
    img_pil = Image.open(".local/image_left.jpg").convert("RGB")
    mask = Image.open(".local/image_left_fg.png")
    mask = mask.split()[-1]

    feats_ex = model.encode(img_pil_ex)

    start = time.perf_counter()
    feats = model.encode(img_pil)
    end = time.perf_counter()

    print(feats.patches.shape, feats.cls.shape)
    print(f"{int((end-start)*1000)} ms.") 

    mask_features = model.encode_mask(mask, feats)
    print("mask features: ", feats.patches.shape, mask_features.shape)

    sim1 = cosine_similarity_pt(feats_ex.cls, feats.cls) # should be low
    sim2 = cosine_similarity_pt(feats.cls, mask_features) # should be high
    print("SIM.: ", sim1, sim2)


# if __name__=="__main__":
#     """
#     https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_inference.ipynb
#     """
#     import time

#     import torch
#     from PIL import Image

#     from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
#     from dinov3.data.transforms import make_classification_eval_transform

#     model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(bpe_path_or_url="tokenizers/bpe_simple_vocab_16e6.txt.gz")

#     img_pil = Image.open(".local/example.jpg").convert("RGB")
    
#     image_preprocess = make_classification_eval_transform()
#     image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).cuda()
#     texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
#     class_names = ["dog", "chair", "bowl", "tupperware"]
#     tokenized_texts_tensor = tokenizer.tokenize(texts).cuda()
#     model = model.cuda()
#     with torch.autocast('cuda', dtype=torch.float16):
#         with torch.no_grad():
#             image_features = model.encode_image(image_tensor)
#             text_features = model.encode_text(tokenized_texts_tensor)

#     start = time.perf_counter()
#     with torch.autocast('cuda', dtype=torch.float16):
#         with torch.no_grad():
#             image_features = model.encode_image(image_tensor)
#             text_features = model.encode_text(tokenized_texts_tensor)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     similarity = (
#         text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
#     )
#     end = time.perf_counter()
#     print(similarity) 
#     print(f"{int((end-start)*1000)} ms.") 
