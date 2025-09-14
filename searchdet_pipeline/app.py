import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.datastructures import UploadFile as StarletteUploadFile, FormData
from PIL import Image
import numpy as np
import cv2

from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.detector_base import DetectorBase, MockDetector

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


def init_detector_mock() -> DetectorBase:
    mask_size = int(os.getenv("DETECTOR_MASK_SIZE", "32"))
    return MockDetector(mask_size=mask_size)

def init_detector_v2() -> DetectorBase:
    detector_params = {
        'mask_backend': 'fastsam',
        "positive_aggregation": "max",
        'dinov3_backbone': "vit7b16",
        "layer": "layer3",
        'pos_as_query_masks': True,
        'vit_pooling': 'cls',
        "loader": "timm",
        "repo_dir": None,
        "max_embedding_size": 1024,
        'device': "cuda",
        'encoder_device': "cuda",
        "use-heatmap-masks": False,
        'half': True,
        'dinov3_ckpt': None,
        'dino_half_precision': False,
        'backbone': "dinov3_vitb16",
        'min_mask_area': 100,
        'smart_rectangle_filter': True,
        'rectangle_bbox_iou_threshold': 0.95,
        'rectangle_straight_line_ratio': 0.8,
        'rectangle_area_ratio_threshold': 0.95,
        'rectangle_angle_tolerance': 10.0,
        'rectangle_side_ratio_threshold': 0.9,
        'perfect_rectangle_iou_threshold': 0.99,
        'rectangle_similarity_iou_threshold': 0.94,
        'square_similarity_iou_threshold': 0.94,
        'rectangle_use_silhouette': True,
        'hole_area_ratio_threshold': 0.03,
        'min_positive_score': 0.4,
        'decision_threshold': 0.65,
        'enable_image_downscaling': True,
        'max_image_size': 512,
        'downscale_quality': 'bilinear',
        'use_fastsam_with_heatmap': True,
        'use_heatmap_sam_hybrid': True,
        'heatmap_sam_threshold': 0.7,
        'sam_refinement_enabled': True,
        'max_hotspots_for_sam': 10,
    }
    return SearchDetDetector(**detector_params)

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

def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)

def mask_to_polygons(mask_2d, min_area: int = 3):
    """
    Convert a 2D binary mask (list-of-lists or ndarray) to polygons using cv2.findContours.
    Returns: List[List[[x, y], ...]] (one polygon = list of [x, y] points).
    Coordinates are in the mask grid space (0..W-1, 0..H-1).

    min_area filters tiny specks; tune as needed.
    """
    if cv2 is None:
        # Fallback: single bounding-rectangle polygon from the mask
        m = np.asarray(mask_2d, dtype=np.uint8)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return []
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

    m = np.asarray(mask_2d, dtype=np.uint8)
    # OpenCV expects 0/255 for binary; ensure it:
    m = (m > 0).astype(np.uint8) * 255

    # Find external contours; you can switch to RETR_TREE if you want holes/hierarchies
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        # Optional simplification (tune epsilon):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.001 * peri, True)
        pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if pts:
            polys.append(pts)
    return polys

def to_python(obj):
    if isinstance(obj, dict):
        return {to_python(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_python(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = init_detector_v2()
    logger.info("Detector initialized")
    yield

app = FastAPI(
    title="Mock Detection Service", 
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(LatencyLoggingMiddleware)
app.add_middleware(AuthMiddleware)

def get_detector(request: Request) -> MockDetector:
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
        image_np = np.array(im, dtype=np.uint8)
        res = detector.find_present_elements(image_np)

        for det in res.get("masks", []):
            seg = None
            if "segmentation" in det:
                seg = det.pop("segmentation")
            elif "mask" in det and isinstance(det["mask"], dict) and "segmentation" in det["mask"]:
                seg = det["mask"].pop("segmentation")

            if seg is not None:
                det["polygons"] = mask_to_polygons(seg)

        if "timing_info" in res:
            del res["timing_info"]
        if "heatmap" in res:
            del res["heatmap"]
        batch_results.append(res)

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
