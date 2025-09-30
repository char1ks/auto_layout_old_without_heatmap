# NOTE: (@gas) reference is Meta example notebook: https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_inference.ipynb
from dataclasses import dataclass
import torch
from PIL import Image

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

# TODO: (@gas) automate instead of hardcode
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


@dataclass
class DinoFeaturesPT:
    cls: torch.Tensor
    patches: torch.Tensor


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

    def resize_transform(self, mask_image: Image.Image) -> torch.Tensor:
        """
        image resize transform to dimensions divisible by patch size
        """
        w, h = mask_image.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        return TF.to_tensor(TF.resize(mask_image, (h_patches * self.patch_size, w_patches * self.patch_size)))

    def encode(self, images: list[Image.Image]) -> DinoFeaturesPT:
        """
        rgb image --> tuple(patch features, cls feature vector)
        shapes:
            (H, W, CH) --> tuple((B, D, H, W), (B, D)) --> __tuple((D, H, W), (,D))__
        """
        # TODO: (@gas) now supports only input images of equal size - fix it somehow | URGENT
        # RuntimeError: stack expects each tensor to be equal size, but got [3, 768, 960] at entry 0 and [3, 768, 768] at entry 1
        resized = torch.stack([self.resize_transform(img) for img in images])
        resized = TF.normalize(resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        resized = resized.cuda()
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = self.model.get_intermediate_layers(
                    resized, 
                    n=range(self.n_layers), 
                    return_class_token=True,
                    reshape=True, 
                    norm=True,
                )[-1] # NOTE: (@gas) get a tuple of (patches, cls)
        return DinoFeaturesPT(cls=feats[1].detach(), patches=feats[0].detach())

    def encode_mask(
        self, 
        masks: list[Image.Image], 
        features: DinoFeaturesPT | None = None, 
        images: list[Image.Image] = [], 
        mask_threshold: float = 0.5,
    ) -> torch.Tensor:
        if features is None and not images:
            raise ValueError("either of features or img_pil should be passed, got none of them")
        resized = torch.stack([self.resize_transform(mask) for mask in masks])
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
        return patches_mean.detach()


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

    # feats = model.encode([img_pil, img_pil_ex])
    # print("Encode batch out shape: ", feats.cls.shape)


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
