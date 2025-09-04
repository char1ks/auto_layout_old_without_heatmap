import gradio as gr
import requests
import io
import os
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple, Any
import base64
import json

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

def draw_polygons_on_image(image: Image.Image, polygons: List[List[List[int]]], 
                          bboxes: List[Tuple[int, int, int, int]] = None, 
                          class_ids: List[str] = None) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    
    for idx, polygon_list in enumerate(polygons):
        color = colors[idx % len(colors)]
        
        # Draw polygons
        for polygon in polygon_list:
            if len(polygon) >= 3:
                points = [(p[0], p[1]) for p in polygon]
                draw.polygon(points, outline=color, width=2, fill=None)
        
        # Draw bounding boxes if provided
        if bboxes and idx < len(bboxes):
            bbox = bboxes[idx]
            draw.rectangle(bbox, outline=color, width=3)
        
        # Draw class labels if provided
        if class_ids and idx < len(class_ids) and bboxes and idx < len(bboxes):
            x1, y1, x2, y2 = bboxes[idx]
            draw.text((x1, y1-20), f"Class: {class_ids[idx]}", fill=color)
    
    return img_copy

# Global client instance
client = APIClient(BASE_URL, USERNAME, PASSWORD)

# Global state for training images
training_state = {
    "images": {},  # {class_id: [cropped_images]}
    "ready_for_inference": False
}

def check_server_status():
    if client.health_check():
        return "✅ Server is running"
    else:
        return "❌ Server is not accessible"

def add_training_image(image, class_id, bbox_data):
    if image is None or not class_id.strip():
        return "Please provide both an image and class ID", None, training_state["images"]
    
    class_id = class_id.strip()
    
    # Convert gradio image format to PIL
    if hasattr(image, 'image'):
        pil_image = image.image
    else:
        pil_image = image
    
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)
    
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
    
    return f"Added image for class '{class_id}'", display_image, summary

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
            return "✅ Training data sent successfully! You can now proceed to inference.", True
        else:
            return "❌ Failed to send training data to server", False
    except Exception as e:
        return f"❌ Error: {str(e)}", False

def clear_training_data():
    training_state["images"].clear()
    training_state["ready_for_inference"] = False
    return "Training data cleared", False, "No training images"

def perform_inference(image):
    if not training_state["ready_for_inference"]:
        return "Please complete the training step first", None
    
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
        masks = result.get("masks", [])
        
        if not masks:
            return "No detections found", pil_image
        
        # Extract polygons, bounding boxes, and class IDs
        all_polygons = []
        all_bboxes = []
        all_class_ids = []
        
        for mask in masks:
            if "polygons" in mask:
                all_polygons.append(mask["polygons"])
            else:
                all_polygons.append([])
            
            if "bbox" in mask:
                bbox = mask["bbox"]
                if len(bbox) == 4:
                    all_bboxes.append(tuple(map(int, bbox)))
                else:
                    all_bboxes.append((0, 0, 10, 10))
            else:
                all_bboxes.append((0, 0, 10, 10))
            
            class_id = mask.get("class_id", mask.get("category_id", "unknown"))
            all_class_ids.append(str(class_id))
        
        # Draw results on image
        result_image = draw_polygons_on_image(
            pil_image, all_polygons, all_bboxes, all_class_ids
        )
        
        # Create result summary
        summary = f"Found {len(masks)} detection(s):\n"
        for i, mask in enumerate(masks):
            class_id = mask.get("class_id", mask.get("category_id", "unknown"))
            score = mask.get("score", mask.get("confidence", "N/A"))
            summary += f"- Detection {i+1}: Class {class_id}, Score: {score}\n"
        
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
            with gr.Column():
                training_image = gr.Image(label="Upload Training Image", type="pil")
                class_id_input = gr.Textbox(label="Class ID", placeholder="Enter class identifier")
                
                gr.Markdown("**Optional: Crop using bounding box**")
                gr.Markdown("Enter bbox coordinates as: x1,y1,x2,y2 (e.g., 100,100,300,300)")
                bbox_input = gr.Textbox(label="Bounding Box (x1,y1,x2,y2)", placeholder="100,100,300,300")
                
                add_btn = gr.Button("Add Training Image", variant="primary")
                
            with gr.Column():
                preview_image = gr.Image(label="Preview (with bbox if specified)", interactive=False)
                add_status = gr.Textbox(label="Status", interactive=False)
                training_summary = gr.Textbox(label="Training Data Summary", interactive=False, lines=5)
        
        with gr.Row():
            send_btn = gr.Button("Send Training Data to Server", variant="primary")
            clear_btn = gr.Button("Clear Training Data", variant="secondary")
            training_status = gr.Textbox(label="Training Status", interactive=False)
            inference_ready = gr.State(False)
        
        def add_with_bbox(image, class_id, bbox_str):
            bbox_data = None
            if bbox_str.strip():
                try:
                    coords = [int(x.strip()) for x in bbox_str.split(",")]
                    if len(coords) == 4:
                        bbox_data = coords
                except ValueError:
                    pass
            return add_training_image(image, class_id, bbox_data)
        
        add_btn.click(
            fn=add_with_bbox,
            inputs=[training_image, class_id_input, bbox_input],
            outputs=[add_status, preview_image, training_summary]
        )
        
        send_btn.click(
            fn=send_training_data,
            outputs=[training_status, inference_ready]
        )
        
        clear_btn.click(
            fn=clear_training_data,
            outputs=[training_summary, inference_ready, training_status]
        )
        
        # Step 2: Inference
        gr.Markdown("## Step 2: Inference")
        
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