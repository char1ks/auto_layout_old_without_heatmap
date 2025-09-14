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

# NOTE: (@gas) for masks only
patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))


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

DINOV3_LOCATION = "/home/synetra/ml_segmentation/vendor/dinov3"


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

    def encode(self, img_pil: Image.Image) -> tuple[tuple[torch.Tensor, ...], ...]:
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
        return (feats[0].squeeze(), feats[1].squeeze())


if __name__=="__main__":
    import time 

    model = DinoV3EncoderGaz()
 
    img_pil = Image.open(".local/example.jpg").convert("RGB")
    feats = model.encode(img_pil)

    start = time.perf_counter()
    feats = model.encode(img_pil)
    end = time.perf_counter()
    print(len(feats), feats[0].shape, feats[1].shape) # 0 - patches, 1 - cls
    print(f"{int((end-start)*1000)} ms.") 


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
