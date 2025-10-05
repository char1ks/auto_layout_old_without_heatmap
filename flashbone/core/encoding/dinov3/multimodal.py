# NOTE: (@gas) reference is Meta example notebook: https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_inference.ipynb  
import torch
from PIL import Image
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from dinov3.data.transforms import make_classification_eval_transform

from flashbone.core.encoding.base import DinoFeaturesPT


class DinoV3VisionTextEncoderGaz:
    def __init__(self):
        self.model, self.tokenizer = \
            dinov3_vitl16_dinotxt_tet1280d20h24l(bpe_path_or_url="tokenizers/bpe_simple_vocab_16e6.txt.gz")
        self.model = self.model.cuda()
        self.preprocessor = make_classification_eval_transform()

    def encode_images(self, images: list[Image.Image]) -> DinoFeaturesPT:
        image_tensor = torch.stack([self.preprocessor(image) for image in images], dim=0).cuda()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                image_features, patch_tokens, backbone_patch_tokens = self.model.encode_image_with_patch_tokens(image_tensor)
        return DinoFeaturesPT(cls=image_features.detach(), patches=patch_tokens.detach())
 
    def encode_texts(self, texts: list[str] = []) -> torch.Tensor:
        tokenized_texts_tensor = self.tokenizer.tokenize(texts).cuda() 
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                text_features = self.model.encode_text(tokenized_texts_tensor)
        return text_features


if __name__=="__main__":
    """
    https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_inference.ipynb
    """
    import time

    import torch
    from PIL import Image

    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
    from dinov3.data.transforms import make_classification_eval_transform

    img_pil = Image.open(".local/example.jpg").convert("RGB")
    texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
    class_names = ["dog", "chair", "bowl", "tupperware"]

    dino_text_encoder = DinoV3VisionTextEncoderGaz()
    image_features = dino_text_encoder.encode_images([img_pil])
    text_features = dino_text_encoder.encode_texts(texts)

    print(image_features.cls.shape, text_features.shape)

    image_features.cls /= image_features.cls.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (
        text_features.cpu().float().numpy() @ image_features.cls.cpu().float().numpy().T
    )
    print(list(zip(texts, similarity)))
