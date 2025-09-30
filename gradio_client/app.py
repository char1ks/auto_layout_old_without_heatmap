import gradio as gr
import requests
import io
import os
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple, Any
import base64
import json
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
    
    def set_references(self, positive_images: Dict[str, List[bytes]], negative_images: List[bytes] = None) -> bool:
        files = []
        
        # Add positive images
        for class_id, images in positive_images.items():
            for idx, img_bytes in enumerate(images):
                field_name = f"positive[{class_id}]"
                files.append((field_name, (f"{class_id}_{idx}.png", img_bytes, "image/png")))
        
        # Add negative images
        if negative_images:
            for idx, img_bytes in enumerate(negative_images):
                files.append(("negative", (f"neg_{idx}.png", img_bytes, "image/png")))
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/set", files=files)
            return response.status_code == 204
        except Exception as e:
            print(f"Error in set_references: {e}")
            return False
    
    def infer(self, images: List[bytes]) -> List[Dict[str, Any]]:
        files = []
        for idx, img_bytes in enumerate(images):
            files.append(("images", (f"img_{idx}.png", img_bytes, "image/png")))
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/infer", files=files)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Inference failed with status {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"Error in inference: {e}")
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

# Global state for training images
training_state = {
    "images": {},  # {class_id: [cropped_images]}
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
    """Add training image using coordinate input"""
    if image is None or not class_id.strip():
        return "Please provide both an image and class ID", None, "No training images", None
    
    class_id = class_id.strip()
    
    # Convert gradio image format to PIL
    if hasattr(image, 'image'):
        pil_image = image.image
    else:
        pil_image = image
    
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)
    
    # Parse bbox coordinates
    bbox_data = None
    if bbox_str.strip():
        try:
            coords = [int(x.strip()) for x in bbox_str.split(",")]
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
            display_image = draw_bbox_on_image(pil_image, (x1, y1, x2, y2))
        else:
            display_image = pil_image
    else:
        display_image = pil_image
    
    # Add to training state
    if class_id not in training_state["images"]:
        training_state["images"][class_id] = []
    
    training_state["images"][class_id].append(cropped_image)
    
    # Create summary text
    summary = "Training Images:\n"
    for cid, imgs in training_state["images"].items():
        summary += f"- Class '{cid}': {len(imgs)} images\n"
    
    # Create training grid
    grid_image = create_training_grid()
    
    return f"Added image for class '{class_id}'", display_image, summary, grid_image

def send_training_data():
    if not training_state["images"]:
        return "No training images to send", False
    
    try:
        # Convert images to bytes
        positive_images = {}
        for class_id, images in training_state["images"].items():
            positive_images[class_id] = [image_to_bytes(img) for img in images]
        
        # Send to server
        success = client.set_references(positive_images)
        
        if success:
            training_state["ready_for_inference"] = True
            return "✅ Training data sent successfully!", True
        else:
            return "❌ Failed to send training data to server", False
    except Exception as e:
        return f"❌ Error: {str(e)}", False

def clear_training_data():
    training_state["images"].clear()
    training_state["ready_for_inference"] = False
    return "Training data cleared", False, "No training images", None, None

def perform_inference(image):
    if image is None:
        return "Please provide an image for inference", None
    
    try:
        # Convert gradio image format to PIL
        if hasattr(image, 'image'):
            pil_image = image.image
        else:
            pil_image = image
            
        if isinstance(pil_image, np.ndarray):
            pil_image = Image.fromarray(pil_image)
        
        # Convert to bytes and send for inference
        img_bytes = image_to_bytes(pil_image)
        results = client.infer([img_bytes])
        
        if not results:
            return "No results returned from server", pil_image

        # Process first result (since we sent one image)
        result = results[0]
        
        # The server now returns DetectionResult objects as dicts
        # Each result is a single DetectionResult with: polygons, bbox, area, score, class_id
        if not result:
            return "No detections found", pil_image
        
        # Handle the new DetectionResult format
        # Result can be either a single DetectionResult or list of DetectionResults
        if isinstance(result, list):
            detections = result
        else:
            detections = [result]
        
        if not detections:
            return "No detections found", pil_image
        
        # Extract polygons, bounding boxes, and class IDs from DetectionResult format
        all_polygons = []
        all_bboxes = []
        all_class_ids = []
        
        for detection in detections:
            # Get polygons (list of list of list of int)
            polygons = detection.get("polygons", [])
            all_polygons.append(polygons)
            
            # Get bbox (list of 4 ints: [x, y, w, h])
            bbox = detection.get("bbox", [0, 0, 10, 10])
            if len(bbox) == 4:
                all_bboxes.append(tuple(map(int, bbox)))
            else:
                all_bboxes.append((0, 0, 10, 10))
            
            # Get class_id
            class_id = detection.get("class_id", "unknown")
            all_class_ids.append(str(class_id))
        
        # Draw results on image (bboxes are in xywh format)
        result_image = draw_polygons_on_image(
            pil_image, all_polygons, all_bboxes, all_class_ids
        )
        
        # Create result summary
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
        return f"Error during inference: {str(e)}", None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="ML Segmentation Client", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ML Segmentation Client")
        gr.Markdown("Two-step pipeline: 1) Add training images with optional cropping, 2) Perform inference")
        
        # Server status
        with gr.Row():
            status_btn = gr.Button("Check Server Status")
            status_text = gr.Textbox(label="Server Status", interactive=False)
        
        status_btn.click(fn=check_server_status, outputs=status_text)
        
        # Step 1: Training
        gr.Markdown("## Step 1: Training Data")
        
        with gr.Row():
            with gr.Column(scale=1):
                training_image = gr.Image(label="Upload Training Image", type="pil")
                class_id_input = gr.Textbox(label="Class ID", placeholder="Enter class identifier")
                
                gr.Markdown("**Optional: Crop using bounding box**")
                gr.Markdown("Enter bbox coordinates as: x1,y1,x2,y2 (e.g., 100,100,300,300)")
                bbox_input = gr.Textbox(label="Bounding Box (x1,y1,x2,y2)", placeholder="100,100,300,300")
                
                add_btn = gr.Button("Add Training Image", variant="primary")
                
            with gr.Column(scale=1):
                preview_image = gr.Image(label="Preview (with bbox if specified)", interactive=False)
                add_status = gr.Textbox(label="Status", interactive=False)
                training_summary = gr.Textbox(label="Training Data Summary", interactive=False, lines=5)
                
            with gr.Column(scale=1):
                training_grid = gr.Image(label="Training Images Grid", interactive=False)
                gr.Markdown("**All images added for training**")
        
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
        
        send_btn.click(
            fn=send_training_data,
            outputs=[training_status, inference_ready]
        )
        
        def clear_training_data():
            training_state["images"].clear()
            training_state["ready_for_inference"] = False
            return "Training data cleared", False, "No training images", None, None
        
        clear_btn.click(
            fn=clear_training_data,
            outputs=[training_status, inference_ready, training_summary, preview_image, training_grid]
        )
        
        # Step 2: Inference
        gr.Markdown("## Step 2: Inference")
        gr.Markdown("💡 **Note**: You can run inference even without training in this session - the server may already have training data loaded from previous sessions.")
        
        with gr.Row():
            with gr.Column():
                inference_image = gr.Image(label="Upload Image for Inference", type="pil")
                infer_btn = gr.Button("Run Inference", variant="primary")
                
            with gr.Column():
                result_image = gr.Image(label="Results (with polygons, bboxes, and class IDs)", interactive=False)
                inference_results = gr.Textbox(label="Inference Results", interactive=False, lines=5)
        
        infer_btn.click(
            fn=perform_inference,
            inputs=[inference_image],
            outputs=[inference_results, result_image]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)