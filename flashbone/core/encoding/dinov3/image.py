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

    # NOTE: (aod) Batching 
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


    # NOTE: (aod)  Мы раньше приводил тензор изначально к CUDA(переменная mask_quantized),
    # а потом насильно к CPU,если не ошибаюсь,то происходила передача данных по PCIe,что жестко замедляло процесс (~40 mS)
    def encode_mask(
        self,
        masks: list[Image.Image],
        features: DinoFeaturesPT | None = None,
        images: list[Image.Image] | None = None,
        mask_threshold: float = 0.5,
    ) -> torch.Tensor:
        if features is None and not images:
            raise ValueError("either features or images must be provided")

        #NOTE: (aod) Тут приводим все маски к единому размеру, преобразуя их в патч-маску под
        #разрешение энкодера и бинаризуем по порогу, получая батч из карт foreground/background патчей.
        target_size = self._compute_target_size(masks)
        resized = torch.stack([self.resize_transform(m, target_size) for m in masks])  # (B, 1, H, W)
        mask_quantized = self.patch_quant_filter(resized)  # (B, 1, P_H, P_W)
        mask_sel = (mask_quantized > mask_threshold).float()

        if features is None:
            features = self.encode(images)

        #NOTE: (aod) блокк ода вычисляет средний эмбеддинг объекта в каждомм изображении, используя векторизированные тензорные операции
        #(это операции ,которые включают в себя умножение, суммирование, деление и выполняются над целым массивом)
        patches = features.patches.float()  # (B, D, P_H, P_W)
        weighted_sum = (patches * mask_sel).sum(dim=(2, 3))  # (B, D),тут получили взвешенную сумму фичей в выделенной области 
        weights = mask_sel.sum(dim=(2, 3)).clamp_min(1e-6)  # (B, 1),считаем, сколько патчей попало в маску
        mean_vec = weighted_sum / weights  # (B, D) ,делим взвешенную сумму на количество активных патчей=>получаем средний вектор признаков объекта для каждого изображения

        return mean_vec.detach().float()



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
