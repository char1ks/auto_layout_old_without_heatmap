import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple
from dataclasses import asdict

from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.datastructures import UploadFile as StarletteUploadFile, FormData
from PIL import Image
import numpy as np
from ultralytics import FastSAM

from flashbone.core.encoding import DinoV3EncoderGaz
from flashbone.core.segmentation import SegmenterConfig, SamSegmenter
from flashbone.core.heatmap_generation import HeatmapGenerator
from flashbone.core.classifier import ClassifierKNN
from flashbone.core.detector import SearchDetDetector
from flashbone.core.detector_base import DetectorBase

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge selected extras
        for key in ("method", "path", "status", "duration_ms", "client", "request_id", "detail"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
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
    heatmap_generator = HeatmapGenerator(dino_fe=encoder, use_cosine_similarity_for_heatmap=False)
    sam = SamSegmenter(
        sam_model=sam_model,
        config=SegmenterConfig(
            min_mask_area=200,
            confidence_threshold=0.5,
            iou_threshold=0.8,
            mask_threshold=0.5,
        )
    )
    classifier = ClassifierKNN(encoder=encoder, d=1024)
    detector = SearchDetDetector(segmenter=sam, classifier=classifier, heatmap_generator=heatmap_generator)
    return detector


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
    logger.info("Detector initialized")
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/set", status_code=204)
async def set_examples(
    request: Request,
    detector: DetectorBase = Depends(get_detector),
):
    """
    Multipart form with:
      positive[<class_id>]: <file...>  (repeat for each class and file)
      negative: <file...>              (optional, can repeat)
    """
    pos_by_class, neg_imgs = await parse_pos_neg_from_form(request)
    detector.set_references(pos_by_class=pos_by_class, neg_imgs=neg_imgs or [])
    return Response(status_code=204)


@app.post("/api/v1/infer")
async def infer(
    images: List[UploadFile] = File(..., description="List of 3-channel images"),
    detector: DetectorBase = Depends(get_detector),
):
    pil_imgs: List[Image.Image] = []
    for f in images:
        data = await f.read()
        try:
            pil_imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            pil_imgs.append(Image.new("RGB", (256, 256)))

    batch_results: List[Dict[str, Any]] = []
    for im in pil_imgs:
        res = detector.find_present_elements(im)
        res_dict = asdict(res)
        batch_results.append(res_dict)

    return JSONResponse(content=jsonable_encoder(batch_results))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
