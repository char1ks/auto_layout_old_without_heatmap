# NOTE: (@gas) this code if FULLY ai-generated
import gradio as gr
import requests
import io
import os
import base64
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple, Any
import math

# Server configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
BASE_URL = f"http://{HOST}:{PORT}"
USERNAME = os.getenv("APP_USERNAME", "admin")
PASSWORD = os.getenv("APP_PASSWORD", "secret")

class APIClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.auth = (username, password)
    
    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def detection_train(self, positive_images: Dict[str, List[bytes]], negative_images: List[bytes] = None) -> bool:
        files = []

        for class_id, images in positive_images.items():
            for idx, img_bytes in enumerate(images):
                field_name = f"positive[{class_id}]"
                files.append((field_name, (f"{class_id}_{idx}.png", img_bytes, "image/png")))

        if negative_images:
            for idx, img_bytes in enumerate(negative_images):
                files.append(("negative", (f"neg_{idx}.png", img_bytes, "image/png")))

        try:
            response = self.session.post(f"{self.base_url}/api/v1/detection/train", files=files)
            return response.status_code == 204
        except Exception as e:
            print(f"Error in detection_train: {e}")
            return False

    def detection_infer(self, images: List[bytes], heatmap_threshold: float = None, class_threshold: float = 0.5) -> List[Dict[str, Any]]:
        files = []
        for idx, img_bytes in enumerate(images):
            files.append(("images", (f"img_{idx}.png", img_bytes, "image/png")))

        data = {}
        if heatmap_threshold is not None:
            data["heatmap_threshold"] = str(heatmap_threshold)
        data["class_threshold"] = str(class_threshold)

        try:
            response = self.session.post(f"{self.base_url}/api/v1/detection/infer", files=files, data=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Detection inference failed with status {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"Error in detection inference: {e}")
            return []

    def classification_train(self, classes: List[Dict[str, Any]]) -> bool:
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/classification/train",
                json={"classes": classes}
            )
            return response.status_code == 204
        except Exception as e:
            print(f"Error in classification_train: {e}")
            return False

    def classification_infer(self, images: List[bytes] = None, texts: List[str] = None, threshold: float = 0.474, topk: int = 1) -> List[Dict[str, Any]]:
        images_b64 = []
        if images:
            for img_bytes in images:
                images_b64.append(base64.b64encode(img_bytes).decode('utf-8'))

        payload = {
            "images": images_b64,
            "texts": texts or [],
            "threshold": threshold,
            "topk": topk
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/classification/infer",
                json=payload
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Classification inference failed with status {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"Error in classification inference: {e}")
            return []

def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))

def draw_bbox_on_image(image: Image.Image, bbox: Tuple[int, int, int, int], color="red", width=3) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(bbox, outline=color, width=width)
    return img_copy

def xywh_to_xyxy(bbox_xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert bbox from xywh format to xyxy format"""
    x, y, w, h = bbox_xywh
    return (x, y, x + w, y + h)

def draw_polygons_on_image(image: Image.Image, polygons: List[List[List[int]]], 
                          bboxes_xywh: List[Tuple[int, int, int, int]] = None, 
                          class_ids: List[str] = None) -> Image.Image:
    img_copy = image.copy().convert('RGBA')
    
    # Create overlay for transparent fills
    overlay = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(img_copy)
    
    colors_rgb = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 165, 0), 
                  (128, 0, 128), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors_str = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    
    for idx, polygon_list in enumerate(polygons):
        color_rgb = colors_rgb[idx % len(colors_rgb)]
        color_str = colors_str[idx % len(colors_str)]
        
        # Draw polygons with transparent fill
        for polygon in polygon_list:
            if len(polygon) >= 3:
                points = [(p[0], p[1]) for p in polygon]
                # Fill with transparent color (30% opacity)
                fill_color = color_rgb + (77,)  # 77 = 30% of 255
                draw_overlay.polygon(points, fill=fill_color, outline=None)
                # Draw outline
                draw_main.polygon(points, outline=color_str, width=2, fill=None)
        
        # Draw bounding boxes if provided (convert from xywh to xyxy)
        if bboxes_xywh and idx < len(bboxes_xywh):
            bbox_xyxy = xywh_to_xyxy(bboxes_xywh[idx])
            draw_main.rectangle(bbox_xyxy, outline=color_str, width=3)
            
            # Draw class labels if provided
            if class_ids and idx < len(class_ids):
                x1, y1, x2, y2 = bbox_xyxy
                draw_main.text((x1, y1-20), f"{class_ids[idx]}", fill=color_str)
    
    # Composite the overlay onto the main image
    img_copy = Image.alpha_composite(img_copy, overlay)
    return img_copy.convert('RGB')

# Global client instance
client = APIClient(BASE_URL, USERNAME, PASSWORD)

training_state = {
    "images": {},
    "texts": {},
    "negative_images": [],
    "negative_texts": [],
    "ready_for_inference": False
}

def create_training_grid() -> Optional[Image.Image]:
    """Create a grid display of all training images"""
    if not training_state["images"]:
        return None
    
    # Collect all images with their class labels
    all_images = []
    for class_id, images in training_state["images"].items():
        for img in images:
            all_images.append((img, class_id))
    
    if not all_images:
        return None
    
    # Calculate grid dimensions
    total_images = len(all_images)
    cols = min(4, total_images)  # Max 4 columns
    rows = math.ceil(total_images / cols)
    
    # Thumbnail size
    thumb_size = 128
    padding = 10
    label_height = 20
    
    # Create grid image
    grid_width = cols * (thumb_size + padding) + padding
    grid_height = rows * (thumb_size + label_height + padding) + padding
    
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    for idx, (img, class_id) in enumerate(all_images):
        row = idx // cols
        col = idx % cols
        
        # Calculate position
        x = col * (thumb_size + padding) + padding
        y = row * (thumb_size + label_height + padding) + padding
        
        # Create thumbnail
        thumb = img.copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        
        # Center the thumbnail in the allocated space
        thumb_w, thumb_h = thumb.size
        thumb_x = x + (thumb_size - thumb_w) // 2
        thumb_y = y + (thumb_size - thumb_h) // 2
        
        # Paste thumbnail
        grid_img.paste(thumb, (thumb_x, thumb_y))
        
        # Add class label
        draw = ImageDraw.Draw(grid_img)
        label_x = x + thumb_size // 2
        label_y = y + thumb_size + 5
        
        # Get text size for centering
        try:
            bbox = draw.textbbox((0, 0), class_id)
            text_width = bbox[2] - bbox[0]
            label_x = label_x - text_width // 2
        except:
            # Fallback for older PIL versions
            pass
        
        draw.text((label_x, label_y), class_id, fill='black')
    
    return grid_img

def check_server_status():
    if client.health_check():
        return "✅ Server is running"
    else:
        return "❌ Server is not accessible"

def add_training_image(image, class_id, bbox_str):
    """Add positive training image using coordinate input"""
    if image is None or not class_id.strip():
        return "Please provide both an image and class ID", None, _create_training_summary(), None

    class_id = class_id.strip()

    # Convert gradio image format to PIL
    if hasattr(image, 'image'):
        pil_image = image.image
    else:
        pil_image = image

    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    # Parse bbox coordinates (tolerate trailing comma)
    bbox_data = None
    if bbox_str.strip():
        try:
            coords = [int(x.strip()) for x in bbox_str.rstrip(', \t\n').split(",") if x.strip()]
            if len(coords) == 4:
                bbox_data = coords
        except ValueError:
            pass

    # If bbox is provided, crop the image
    cropped_image = pil_image
    if bbox_data and len(bbox_data) == 4:
        x1, y1, x2, y2 = bbox_data
        if x1 < x2 and y1 < y2:
            cropped_image = crop_image(pil_image, (x1, y1, x2, y2))
            display_image = cropped_image  # Show cropped result
        else:
            display_image = pil_image
    else:
        display_image = pil_image

    # Add to training state
    if class_id not in training_state["images"]:
        training_state["images"][class_id] = []

    training_state["images"][class_id].append(cropped_image)

    # Create training grid
    grid_image = create_training_grid()

    return f"Added positive image for class '{class_id}'", display_image, _create_training_summary(), grid_image

def add_negative_image(image, bbox_str):
    """Add negative training image using coordinate input"""
    if image is None:
        return "Please provide an image", None, _create_training_summary()

    # Convert gradio image format to PIL
    if hasattr(image, 'image'):
        pil_image = image.image
    else:
        pil_image = image

    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    # Parse bbox coordinates (tolerate trailing comma)
    bbox_data = None
    if bbox_str.strip():
        try:
            coords = [int(x.strip()) for x in bbox_str.rstrip(', \t\n').split(",") if x.strip()]
            if len(coords) == 4:
                bbox_data = coords
        except ValueError:
            pass

    # If bbox is provided, crop the image
    cropped_image = pil_image
    if bbox_data and len(bbox_data) == 4:
        x1, y1, x2, y2 = bbox_data
        if x1 < x2 and y1 < y2:
            cropped_image = crop_image(pil_image, (x1, y1, x2, y2))
            display_image = cropped_image  # Show cropped result
        else:
            display_image = pil_image
    else:
        display_image = pil_image

    # Add to negative training state
    training_state["negative_images"].append(cropped_image)

    return f"Added negative image ({len(training_state['negative_images'])} total)", display_image, _create_training_summary()

def _create_training_summary():
    summary = "Training Data:\n"
    for cid, imgs in training_state["images"].items():
        summary += f"- Class '{cid}': {len(imgs)} images"
        if cid in training_state["texts"]:
            summary += f", {len(training_state['texts'][cid])} texts"
        summary += "\n"
    if training_state["negative_images"]:
        summary += f"- Negative: {len(training_state['negative_images'])} images\n"
    if training_state["negative_texts"]:
        summary += f"- Negative: {len(training_state['negative_texts'])} texts\n"
    if not training_state["images"] and not training_state["texts"] and not training_state["negative_images"] and not training_state["negative_texts"]:
        summary = "No training data"
    return summary

def send_training_data(task_type):
    if task_type == "Detection":
        if not training_state["images"] and not training_state["negative_images"]:
            return "No training images to send", False

        try:
            positive_images = {}
            for class_id, images in training_state["images"].items():
                positive_images[class_id] = [image_to_bytes(img) for img in images]

            negative_images = [image_to_bytes(img) for img in training_state["negative_images"]] if training_state["negative_images"] else None

            success = client.detection_train(positive_images, negative_images)

            if success:
                training_state["ready_for_inference"] = True
                return "✅ Detection training data sent successfully!", True
            else:
                return "❌ Failed to send detection training data", False
        except Exception as e:
            return f"❌ Error: {str(e)}", False

    else:  # Classification
        if not training_state["images"] and not training_state["texts"]:
            return "No training data to send", False

        try:
            classes = []
            all_class_ids = set(training_state["images"].keys()) | set(training_state["texts"].keys())

            for class_id in all_class_ids:
                images_b64 = []
                if class_id in training_state["images"]:
                    for img in training_state["images"][class_id]:
                        img_bytes = image_to_bytes(img)
                        images_b64.append(base64.b64encode(img_bytes).decode('utf-8'))

                texts = training_state["texts"].get(class_id, [])

                negative_images_b64 = []
                if training_state["negative_images"]:
                    for img in training_state["negative_images"]:
                        img_bytes = image_to_bytes(img)
                        negative_images_b64.append(base64.b64encode(img_bytes).decode('utf-8'))

                classes.append({
                    "class_id": int(class_id) if class_id.isdigit() else 0,
                    "images": images_b64,
                    "texts": texts,
                    "negative_images": negative_images_b64,
                    "negative_texts": training_state["negative_texts"]
                })

            success = client.classification_train(classes)

            if success:
                training_state["ready_for_inference"] = True
                return "✅ Classification training data sent successfully!", True
            else:
                return "❌ Failed to send classification training data", False
        except Exception as e:
            return f"❌ Error: {str(e)}", False

def clear_training_data():
    training_state["images"].clear()
    training_state["texts"].clear()
    training_state["negative_images"].clear()
    training_state["negative_texts"].clear()
    training_state["ready_for_inference"] = False
    return "Training data cleared", False, "No training data", None, None

def perform_inference(task_type, image, heatmap_threshold, class_threshold):
    if task_type == "Detection":
        if image is None:
            return "Please provide an image for detection", None

        try:
            if hasattr(image, 'image'):
                pil_image = image.image
            else:
                pil_image = image

            if isinstance(pil_image, np.ndarray):
                pil_image = Image.fromarray(pil_image)

            img_bytes = image_to_bytes(pil_image)

            heatmap_thresh = None if heatmap_threshold == "" or heatmap_threshold is None else float(heatmap_threshold)
            class_thresh = 0.5 if class_threshold == "" or class_threshold is None else float(class_threshold)

            results = client.detection_infer([img_bytes], heatmap_threshold=heatmap_thresh, class_threshold=class_thresh)

            if not results:
                return "No results returned from server", pil_image

            result = results[0]

            if not result:
                return "No detections found", pil_image

            if isinstance(result, list):
                detections = result
            else:
                detections = [result]

            if not detections:
                return "No detections found", pil_image

            all_polygons = []
            all_bboxes = []
            all_class_ids = []

            for detection in detections:
                polygons = detection.get("polygons", [])
                all_polygons.append(polygons)

                bbox = detection.get("bbox", [0, 0, 10, 10])
                if len(bbox) == 4:
                    all_bboxes.append(tuple(map(int, bbox)))
                else:
                    all_bboxes.append((0, 0, 10, 10))

                class_id = detection.get("class_id", "unknown")
                all_class_ids.append(str(class_id))

            result_image = draw_polygons_on_image(
                pil_image, all_polygons, all_bboxes, all_class_ids
            )

            summary = f"Found {len(detections)} detection(s):\n"
            for i, detection in enumerate(detections):
                class_id = detection.get("class_id", "unknown")
                score = detection.get("score", "N/A")
                bbox = detection.get("bbox", [])
                area = detection.get("area", "N/A")
                bbox_str = f"[{','.join(map(str, bbox))}]" if bbox else "N/A"

                summary += f"- Detection {i+1}:\n"
                summary += f"  Class: {class_id}\n"
                summary += f"  Score: {score}\n"
                summary += f"  Area: {area}\n"
                summary += f"  Bbox (xywh): {bbox_str}\n"

            return summary, result_image

        except Exception as e:
            return f"Error during detection: {str(e)}", None

    else:  # Classification
        if image is None:
            return "Please provide an image for classification", None

        try:
            pil_image = None

            if hasattr(image, 'image'):
                pil_image = image.image
            else:
                pil_image = image

            if isinstance(pil_image, np.ndarray):
                pil_image = Image.fromarray(pil_image)

            img_bytes = image_to_bytes(pil_image)

            threshold = 0.474 if class_threshold == "" or class_threshold is None else float(class_threshold)

            results = client.classification_infer(
                images=[img_bytes],
                texts=None,
                threshold=threshold,
                topk=1
            )

            if not results:
                return "No classification results returned", pil_image

            summary = f"Found {len(results)} classification(s):\n"
            for i, pred in enumerate(results):
                class_id = pred.get("class_id", "unknown")
                score = pred.get("score", "N/A")
                summary += f"- Prediction {i+1}: Class {class_id}, Score: {score}\n"

            return summary, pil_image

        except Exception as e:
            return f"Error during classification: {str(e)}", None

def add_text_data(class_id, texts_input):
    if not class_id.strip() or not texts_input.strip():
        return "Please provide both class ID and texts", _create_training_summary()

    class_id = class_id.strip()
    texts = [t.strip() for t in texts_input.split(",") if t.strip()]

    if class_id not in training_state["texts"]:
        training_state["texts"][class_id] = []

    training_state["texts"][class_id].extend(texts)

    return f"Added {len(texts)} text(s) for class '{class_id}'", _create_training_summary()


def add_negative_text_data(texts_input):
    if not texts_input.strip():
        return "Please provide negative texts", _create_training_summary()

    texts = [t.strip() for t in texts_input.split(",") if t.strip()]
    training_state["negative_texts"].extend(texts)

    return f"Added {len(texts)} negative text(s)", _create_training_summary()

def create_interface():
    with gr.Blocks(title="ML Detection & Classification Client", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ML Detection & Classification Client")
        gr.Markdown("Two-step pipeline: 1) Add training data, 2) Perform inference")
        
        with gr.Row():
            status_btn = gr.Button("Check Server Status")
            status_text = gr.Textbox(label="Server Status", interactive=False)

        status_btn.click(fn=check_server_status, outputs=status_text)

        with gr.Row():
            task_selector = gr.Radio(
                choices=["Detection", "Classification"],
                value="Detection",
                label="Task Type"
            )

        gr.Markdown("## Step 1: Training Data")

        gr.Markdown("### Positive Images")
        with gr.Row():
            with gr.Column(scale=1):
                training_image = gr.Image(label="Upload Positive Training Image", type="pil", show_label=True)
                coords_display = gr.Textbox(label="Mouse Coordinates (x, y)", value="", interactive=False)
                class_id_input = gr.Textbox(label="Class ID", placeholder="Enter class identifier")

                gr.Markdown("**Optional: Crop using bounding box**")
                gr.Markdown("Enter bbox coordinates as: x1,y1,x2,y2 (e.g., 100,100,300,300)")
                bbox_input = gr.Textbox(label="Bounding Box (x1,y1,x2,y2)", placeholder="100,100,300,300")

                add_btn = gr.Button("Add Positive Image", variant="primary")

            with gr.Column(scale=1):
                preview_image = gr.Image(label="Preview (with bbox if specified)", interactive=False)
                add_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                training_grid = gr.Image(label="Training Images Grid", interactive=False)
                gr.Markdown("**All positive images added**")

        text_section = gr.Group(visible=False)
        with text_section:
            gr.Markdown("### Text Data (Classification only)")
            with gr.Row():
                with gr.Column(scale=1):
                    text_class_id_input = gr.Textbox(label="Class ID", placeholder="Enter class identifier")
                    texts_input = gr.Textbox(label="Texts (comma-separated)", placeholder="text1, text2, text3")
                    add_text_btn = gr.Button("Add Texts", variant="secondary")

                with gr.Column(scale=1):
                    text_add_status = gr.Textbox(label="Status", interactive=False)

        negative_section = gr.Group()
        with negative_section:
            gr.Markdown("### Negative Images")
            with gr.Row():
                with gr.Column(scale=1):
                    negative_image = gr.Image(label="Upload Negative Training Image", type="pil")
                    negative_coords_display = gr.Textbox(label="Mouse Coordinates (x, y)", value="", interactive=False)

                    gr.Markdown("**Optional: Crop using bounding box**")
                    gr.Markdown("Enter bbox coordinates as: x1,y1,x2,y2 (e.g., 100,100,300,300)")
                    negative_bbox_input = gr.Textbox(label="Bounding Box (x1,y1,x2,y2)", placeholder="100,100,300,300")

                    add_negative_btn = gr.Button("Add Negative Image", variant="secondary")

                with gr.Column(scale=1):
                    negative_preview_image = gr.Image(label="Negative Preview (with bbox if specified)", interactive=False)
                    negative_add_status = gr.Textbox(label="Status", interactive=False)

        negative_text_section = gr.Group(visible=False)
        with negative_text_section:
            gr.Markdown("### Negative Text Data (Classification only)")
            with gr.Row():
                with gr.Column(scale=1):
                    negative_texts_input = gr.Textbox(label="Negative Texts (comma-separated)", placeholder="text1, text2, text3")
                    add_negative_text_btn = gr.Button("Add Negative Texts", variant="secondary")

                with gr.Column(scale=1):
                    negative_text_add_status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            training_summary = gr.Textbox(label="Training Data Summary", interactive=False, lines=5)

        with gr.Row():
            send_btn = gr.Button("Send Training Data to Server", variant="primary")
            clear_btn = gr.Button("Clear Training Data", variant="secondary")
            training_status = gr.Textbox(label="Training Status", interactive=False)
            inference_ready = gr.State(False)

        add_btn.click(
            fn=add_training_image,
            inputs=[training_image, class_id_input, bbox_input],
            outputs=[add_status, preview_image, training_summary, training_grid]
        )

        add_text_btn.click(
            fn=add_text_data,
            inputs=[text_class_id_input, texts_input],
            outputs=[text_add_status, training_summary]
        )

        add_negative_btn.click(
            fn=add_negative_image,
            inputs=[negative_image, negative_bbox_input],
            outputs=[negative_add_status, negative_preview_image, training_summary]
        )

        add_negative_text_btn.click(
            fn=add_negative_text_data,
            inputs=[negative_texts_input],
            outputs=[negative_text_add_status, training_summary]
        )

        send_btn.click(
            fn=send_training_data,
            inputs=[task_selector],
            outputs=[training_status, inference_ready]
        )

        def clear_training_data_wrapper():
            training_state["images"].clear()
            training_state["texts"].clear()
            training_state["negative_images"].clear()
            training_state["negative_texts"].clear()
            training_state["ready_for_inference"] = False
            return "Training data cleared", False, "No training data", None, None, None, None, None, None

        clear_btn.click(
            fn=clear_training_data_wrapper,
            outputs=[training_status, inference_ready, training_summary, preview_image, training_grid, negative_preview_image, negative_add_status, text_add_status, negative_text_add_status]
        )
        
        gr.Markdown("## Step 2: Inference")
        gr.Markdown("💡 **Note**: You can run inference even without training in this session - the server may have training data from previous sessions.")

        with gr.Row():
            with gr.Column():
                heatmap_threshold_section = gr.Group()
                with heatmap_threshold_section:
                    heatmap_threshold_input = gr.Number(
                        label="Heatmap Threshold (Detection only, optional)",
                        value=None,
                        placeholder="Leave empty for default",
                        precision=2
                    )

                class_threshold_input = gr.Number(
                    label="Class/Classification Threshold",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    precision=2
                )
                inference_image = gr.Image(label="Upload Image for Inference", type="pil")

                infer_btn = gr.Button("Run Inference", variant="primary")

            with gr.Column():
                result_image = gr.Image(label="Results", interactive=False)
                inference_results = gr.Textbox(label="Inference Results", interactive=False, lines=5)

        def toggle_sections(task_type):
            if task_type == "Detection":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

        task_selector.change(
            fn=toggle_sections,
            inputs=[task_selector],
            outputs=[text_section, negative_section, negative_text_section, heatmap_threshold_section]
        )

        infer_btn.click(
            fn=perform_inference,
            inputs=[task_selector, inference_image, heatmap_threshold_input, class_threshold_input],
            outputs=[inference_results, result_image]
        )

        # Event handlers for mouse coordinates on select (click)
        def show_coords(evt: gr.SelectData):
            if evt.index is not None:
                x, y = evt.index
                return f"x={x}, y={y}"
            return ""

        def update_bbox_coords(current_bbox: str, evt: gr.SelectData):
            """Update bbox coordinates based on click. Reset after two pairs."""
            if evt.index is None:
                return current_bbox, f"x={evt.index[0]}, y={evt.index[1]}" if evt.index else ""

            x, y = evt.index
            coords_display = f"x={x}, y={y}"

            # Parse existing coordinates
            current_bbox = current_bbox.strip()
            if not current_bbox:
                coords = []
            else:
                # Remove trailing comma for parsing
                bbox_clean = current_bbox.rstrip(', \t\n')
                coords = [c.strip() for c in bbox_clean.split(',') if c.strip()]

            # Count existing coordinate values
            if len(coords) == 0:
                # First click - start new bbox
                new_bbox = f"{x},{y},"
            elif len(coords) >= 4:
                # Already have two pairs - reset and start fresh
                new_bbox = f"{x},{y},"
            else:
                # Add to existing coordinates
                new_bbox = ','.join(coords) + f",{x},{y},"

            # Clean trailing comma only if we have exactly 4 coordinates
            coords_final = [c.strip() for c in new_bbox.rstrip(',').split(',') if c.strip()]
            if len(coords_final) == 4:
                new_bbox = ','.join(coords_final)

            return new_bbox, coords_display

        training_image.select(
            fn=update_bbox_coords,
            inputs=[bbox_input],
            outputs=[bbox_input, coords_display]
        )
        negative_image.select(
            fn=update_bbox_coords,
            inputs=[negative_bbox_input],
            outputs=[negative_bbox_input, negative_coords_display]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)