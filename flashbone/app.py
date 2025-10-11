# NOTE: (@gas) this code if __almost fully__ ai-generated (except the detector init)
import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple
from dataclasses import asdict

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.datastructures import UploadFile as StarletteUploadFile, FormData
from pydantic import BaseModel, Field
from PIL import Image
from ultralytics import FastSAM

from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz
from flashbone.core.encoding.dinov3.multimodal import DinoV3VisionTextEncoderGaz
from flashbone.core.segmentation import SegmenterConfig, SamSegmenter
from flashbone.core.heatmap_generation import HeatmapGenerator
from flashbone.core.classification.mask_classifier_knn import MaskClassifierKNN
from flashbone.core.classification.vision_text_classifier_knn import VisionTextClassifierKNN
from flashbone.core.classification.base import ClassData, ClassifierPredictRequest, ClassifierBase
from flashbone.core.detection.searchdet_detector import SearchDetDetector
from flashbone.core.detection.base import DetectorBase
from flashbone.core.image_resizing import ImageResizer


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge selected extras
        for key in ("method", "path", "status", "duration_ms", "client", "request_id", "detail", "inference_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO or os.environ.get("LOG_LEVEL"))
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.handlers = [_handler]
logger.propagate = False

APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "secret")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only protect API routes, allow docs/health
        if request.url.path.startswith("/api/"):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Basic "):
                return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})
            try:
                b64 = auth.split(" ", 1)[1]
                decoded = base64.b64decode(b64).decode("utf-8")
                username, password = decoded.split(":", 1)
            except Exception:
                return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})
            if not (username == APP_USERNAME and password == APP_PASSWORD):
                return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})
        return await call_next(request)


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if response is not None:
                response.headers["X-Server-Latency-ms"] = str(duration_ms)
            logger.info(
                f"{request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": getattr(response, "status_code", None),
                    "duration_ms": duration_ms,
                    "client": request.client.host if request.client else None,
                },
            )


def init_detector() -> DetectorBase:
    sam_model = FastSAM('FastSAM-x.pt')
    encoder = DinoV3EncoderGaz()
    heatmap_generator = HeatmapGenerator(
        dino_fe=encoder,
        use_cosine_similarity_for_heatmap=True,
        threshold_cosine=0.4,
    )
    sam = SamSegmenter(
        sam_model=sam_model,
        config=SegmenterConfig(
            min_mask_area=200,
            confidence_threshold=0.5,
            iou_threshold=0.8,
            mask_threshold=0.5,
        )
    )
    classifier = MaskClassifierKNN(encoder=encoder, d=1024)
    image_resizer = ImageResizer(max_side=1024)
    detector = SearchDetDetector(
        segmenter=sam,
        classifier=classifier,
        heatmap_generator=heatmap_generator,
        image_resizer=image_resizer,
    )
    return detector


def init_vision_text_classifier() -> ClassifierBase:
    encoder = DinoV3VisionTextEncoderGaz()
    classifier = VisionTextClassifierKNN(encoder=encoder, d=2048)
    return classifier


async def parse_infer_form(request: Request) -> Tuple[List[Image.Image], float, float]:
    """
    Parse form data for inference endpoint.
    
    Returns:
        (images, heatmap_threshold, class_threshold)
    """
    form: FormData = await request.form()
    
    # Parse threshold parameters
    heatmap_threshold = None
    class_threshold = 0.5  # Default value
    
    if "heatmap_threshold" in form:
        try:
            heatmap_threshold = float(form["heatmap_threshold"])
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid heatmap_threshold value")
    
    if "class_threshold" in form:
        try:
            class_threshold = float(form["class_threshold"])
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid class_threshold value")
    
    # Parse images
    pil_imgs: List[Image.Image] = []
    if "images" not in form:
        raise HTTPException(status_code=400, detail="No images found in form data")
    
    images = form.getlist("images")
    for f in images:
        if isinstance(f, StarletteUploadFile):
            data = await f.read()
            try:
                pil_imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception:
                pil_imgs.append(Image.new("RGB", (256, 256)))
    
    return pil_imgs, heatmap_threshold, class_threshold


async def parse_pos_neg_from_form(request: Request) -> Tuple[Dict[str, List[Image.Image]], List[Image.Image]]:
    """
    Multipart form format:

      positive[<class_id>]: <file1>
      positive[<class_id>]: <file2>
      positive[<another_class>]: <file3>

      negative: <fileA>
      negative: <fileB>

    Returns:
      (pos_by_class, neg_imgs)
        pos_by_class: Dict[str, List[PIL.Image]]
        neg_imgs: List[PIL.Image]
    """
    form: FormData = await request.form()
    pos_by_class: Dict[str, List[Image.Image]] = {}
    neg_imgs: List[Image.Image] = []

    # Parse positives
    for key in form.keys():
        if key.startswith("positive[") and key.endswith("]"):
            class_id = key[len("positive["):-1]
            items = form.getlist(key)
            for item in items:
                if isinstance(item, StarletteUploadFile):
                    content = await item.read()
                    try:
                        img = Image.open(io.BytesIO(content)).convert("RGB")
                    except Exception:
                        img = Image.new("RGB", (256, 256))
                    pos_by_class.setdefault(class_id, []).append(img)

    # Parse negatives (if provided)
    if "negative" in form:
        items = form.getlist("negative")
        for item in items:
            if isinstance(item, StarletteUploadFile):
                content = await item.read()
                try:
                    img = Image.open(io.BytesIO(content)).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (256, 256))
                neg_imgs.append(img)

    # If nothing at all provided, error; else allow empty negatives
    if not pos_by_class and not neg_imgs:
        raise HTTPException(status_code=400, detail="No images found. Use fields like positive[<class_id>] and optional negative.")

    return pos_by_class, neg_imgs


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = init_detector()
    app.state.vision_text_classifier = init_vision_text_classifier()
    logger.info("Detector and classifier initialized")
    yield


app = FastAPI(
    title="Mock Detection Service", 
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(LatencyLoggingMiddleware)
app.add_middleware(AuthMiddleware)


def get_detector(request: Request) -> DetectorBase:
    return request.app.state.detector


def get_vision_text_classifier(request: Request) -> ClassifierBase:
    return request.app.state.vision_text_classifier


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/detection/train", status_code=204)
async def detection_train(
    request: Request,
    detector: DetectorBase = Depends(get_detector),
):
    pos_by_class, neg_imgs = await parse_pos_neg_from_form(request)
    detector.set_references(pos_by_class=pos_by_class, neg_imgs=neg_imgs or [])
    return Response(status_code=204)


@app.post("/api/v1/detection/infer")
async def detection_infer(
    request: Request,
    detector: DetectorBase = Depends(get_detector),
):
    pil_imgs, heatmap_threshold, class_threshold = await parse_infer_form(request)

    mean_latency_ms: float = 0.0
    batch_results: List[Dict[str, Any]] = []
    for im in pil_imgs:
        start = time.perf_counter()
        res = detector.detect(im, heatmap_threshold=heatmap_threshold, class_threshold=class_threshold)
        res_dict = [asdict(r) for r in res]
        end = time.perf_counter()
        dt = (end - start) * 1000
        mean_latency_ms += dt / len(pil_imgs)
        batch_results.append(res_dict)

    logger.info("Inference completed", extra={"inference_ms": round(mean_latency_ms, 2)})

    return JSONResponse(content=jsonable_encoder(batch_results))


class ClassificationTrainRequest(BaseModel):
    classes: List[Dict[str, Any]] = Field(description="List of class data")

    class Config:
        json_schema_extra = {
            "example": {
                "classes": [
                    {
                        "class_id": 0,
                        "images": ["base64_image_1", "base64_image_2"],
                        "texts": ["donkey"],
                        "negative_images": ["base64_neg_image"],
                        "negative_texts": ["green grass", "blue sky"]
                    }
                ]
            }
        }


class ClassificationInferRequest(BaseModel):
    images: List[str] = Field(default_factory=list, description="Base64 encoded images")
    texts: List[str] = Field(default_factory=list, description="Text queries")
    threshold: float = Field(default=0.474, description="Classification threshold")
    topk: int = Field(default=1, description="Top K results")


@app.post("/api/v1/classification/train", status_code=204)
async def classification_train(
    request: Request,
    body: ClassificationTrainRequest,
    classifier: ClassifierBase = Depends(get_vision_text_classifier),
):
    dataset: List[ClassData] = []
    for cls_data in body.classes:
        images = []
        for img_b64 in cls_data.get("images", []):
            try:
                img_bytes = base64.b64decode(img_b64)
                images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image data")

        negative_images = []
        for img_b64 in cls_data.get("negative_images", []):
            try:
                img_bytes = base64.b64decode(img_b64)
                negative_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid negative image data")

        dataset.append(ClassData(
            class_id=cls_data["class_id"],
            images=images,
            texts=cls_data.get("texts", []),
            negative_images=negative_images,
            negative_texts=cls_data.get("negative_texts", []),
        ))

    classifier.index(dataset)
    return Response(status_code=204)


@app.post("/api/v1/classification/infer")
async def classification_infer(
    request: Request,
    body: ClassificationInferRequest,
    classifier: ClassifierBase = Depends(get_vision_text_classifier),
):
    images = []
    for img_b64 in body.images:
        try:
            img_bytes = base64.b64decode(img_b64)
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image data")

    req = ClassifierPredictRequest(
        images=images,
        texts=body.texts,
        threshold=body.threshold,
        topk=body.topk,
    )

    start = time.perf_counter()
    preds = classifier.predict(req)
    dt = (time.perf_counter() - start) * 1000

    logger.info("Classification inference completed", extra={"inference_ms": round(dt, 2)})

    return JSONResponse(content=jsonable_encoder([{"class_id": int(p.class_id), "score": float(p.score)} for p in preds]))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
