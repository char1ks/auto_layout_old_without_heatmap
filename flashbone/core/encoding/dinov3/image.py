# NOTE: (@gas) reference is Meta example notebook: https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_inference.ipynb
import math

import torch
from PIL import Image
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
import torchvision.transforms.functional as TF

from flashbone.core.encoding.base import (
    MODEL_DINOV3_VITL, 
    PATCH_SIZE, 
    DINOV3_LOCATION, 
    MODEL_TO_NUM_LAYERS, 
    IMAGENET_MEAN, 
    IMAGENET_STD,
    DinoFeaturesPT, 
)


class DinoV3EncoderGaz:
    def __init__(self, model_name: str = MODEL_DINOV3_VITL, image_size: int = 768, patch_size: int = PATCH_SIZE):
        self.n_layers = MODEL_TO_NUM_LAYERS[model_name]
        self.model = torch.hub.load(
            repo_or_dir=DINOV3_LOCATION,
            model=model_name,
            source="local",
        )
        self.model.eval()
        self.model.cuda()
        self.image_size = image_size
        self.patch_size = patch_size
        # NOTE: (@gas) for masks only
        self.patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
        self.patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

    def _compute_target_size(self, images: list[Image.Image]) -> tuple[int, int]:
        h_target = math.ceil(self.image_size / self.patch_size) * self.patch_size
        widths = []
        for img in images:
            w, h = img.size
            new_w = int(round(w * (h_target / h)))
            widths.append(new_w)
        w_target = math.ceil(max(widths) / self.patch_size) * self.patch_size
        return h_target, w_target

    def resize_transform(self, img: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
        """
        image resize transform to dimensions divisible by patch size
        """
        h_target, w_target = target_size
        w, h = img.size
        new_w = int(round(w * (h_target / h)))
        resized = TF.resize(img, (h_target, new_w))
        pad_w = w_target - new_w
        if pad_w > 0:
            resized = TF.pad(resized, [0, 0, pad_w, 0], fill=0)
        return TF.to_tensor(resized)

    def encode(self, images: list[Image.Image]) -> DinoFeaturesPT:
        """
        rgb image --> tuple(patch features, cls feature vector)
        shapes:
            (H, W, CH) --> tuple((B, D, H, W), (B, D)) --> __tuple((D, H, W), (,D))__
        """
        target_size = self._compute_target_size(images)
        resized = torch.stack([self.resize_transform(img, target_size) for img in images])
        resized = TF.normalize(resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        resized = resized.cuda()
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                feats = self.model.get_intermediate_layers(
                    resized, 
                    n=range(self.n_layers), 
                    return_class_token=True,
                    reshape=True, 
                    norm=True,
                )[-1] # NOTE: (@gas) get a tuple of (patches, cls)
        return DinoFeaturesPT(cls=feats[1].detach().float(), patches=feats[0].detach().float())

    def encode_mask(
        self, 
        masks: list[Image.Image], 
        features: DinoFeaturesPT | None = None, 
        images: list[Image.Image] = [], 
        mask_threshold: float = 0.5,
    ) -> torch.Tensor:
        if features is None and not images:
            raise ValueError("either of features or img_pil should be passed, got none of them")
        target_size = self._compute_target_size(masks)
        resized = torch.stack([self.resize_transform(mask, target_size) for mask in masks])
        mask_quantized = self.patch_quant_filter(resized).detach().cpu()
        if features is None and images:
            features = self.encode(images)
        B = len(masks)
        patches_fg_selection = mask_quantized > mask_threshold # (1, 1, P_H, P_W); bool
        patches_fg_selection = patches_fg_selection.view(B, -1)  # [B, N_points]
        patches_selected = []
        patches_mean = []
        for b in range(B):
            indices = patches_fg_selection[b].nonzero(as_tuple=False).squeeze(-1)  # [N_points_b]
            selected = features.patches[b, :, indices // mask_quantized.shape[3], indices % mask_quantized.shape[3]]  # [D, N_points_b]
            mean = selected.mean(dim=1)  # [D]
            patches_selected.append(selected)
            patches_mean.append(mean)
        patches_mean = torch.stack(patches_mean)
        return patches_mean.detach().float()


if __name__=="__main__":
    # TODO: (@gas) convert to tests

    """
    PYTHONPATH=. python flashbone/core/encoding.py
    """
    import time 

    model = DinoV3EncoderGaz()
 
    img_pil_ex = Image.open(".local/example.jpg").convert("RGB")
    # warmup
    feats_ex = model.encode([img_pil_ex])

    img_pil = Image.open(".local/image_left.jpg").convert("RGB")
    mask = Image.open(".local/image_left_fg.png")
    mask = mask.split()[-1]

    start = time.perf_counter()
    feats = model.encode([img_pil])
    end = time.perf_counter()

    print(feats.patches.shape, feats.cls.shape)
    print(f"{int((end-start)*1000)} ms.") 

    mask_features = model.encode_mask([mask], feats)
    print(mask_features.shape)
    # print("mask features: ", feats.patches.shape, mask_features.shape)

    sim1 = torch.cosine_similarity(feats_ex.cls, feats.cls, dim=-1) # should be low
    sim2 = torch.cosine_similarity(feats.cls, mask_features, dim=-1) # should be high
    print("SIM.: ", sim1, sim2)

    feats = model.encode([img_pil, img_pil_ex])
    print("Encode batch out shape: ", feats.cls.shape)
