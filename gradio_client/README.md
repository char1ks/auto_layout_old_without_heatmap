# ML Segmentation Gradio Client

A simple web interface for the ML segmentation API server using Gradio.

## Features

- **Two-step pipeline**:
  1. **Training**: Upload images, optionally crop them with bounding boxes, assign class IDs, and send to server
  2. **Inference**: Upload an image and get back polygon masks, bounding boxes, and class predictions

- **Interactive cropping**: Specify bounding box coordinates to crop training images
- **Visual results**: Inference results are overlaid on the input image with polygons, bboxes, and class labels
- **Server status monitoring**: Check if the API server is accessible

## Setup

1. **Install dependencies**:
   ```bash
   cd gradio_client
   pip install -r requirements.txt
   ```

2. **Configure server connection** (optional):
   The client connects to `http://127.0.0.1:8000` by default. To change this:
   ```bash
   export HOST=your-server-host
   export PORT=your-server-port
   export APP_USERNAME=admin
   export APP_PASSWORD=secret
   ```

## Usage

### Start the client:
```bash
cd gradio_client
python run.py
```

Or directly:
```bash
python app.py
```

The web interface will be available at http://localhost:7860

### Using the Interface

#### Step 1: Training Data
1. Upload training images using the image uploader
2. Enter a class ID for each image (e.g., "cat", "dog", "car")
3. **Optional**: Crop images by specifying bounding box coordinates (format: x1,y1,x2,y2)
4. Click "Add Training Image" to add each image to the training set
5. Review the training summary
6. Click "Send Training Data to Server" to train the model

#### Step 2: Inference
1. Upload an image for inference
2. Click "Run Inference"
3. View the results overlaid on your image with:
   - Polygon masks (colored outlines)
   - Bounding boxes (rectangles)
   - Class labels and confidence scores

## API Server Requirements

This client requires the ML segmentation API server to be running with the following endpoints:
- `GET /health` - Health check
- `POST /api/v1/set` - Set training/reference images
- `POST /api/v1/infer` - Run inference on images

Make sure the API server is started before using this client.

## Troubleshooting

- **"Server is not accessible"**: Make sure the API server is running and accessible at the configured host/port
- **Authentication errors**: Check that APP_USERNAME and APP_PASSWORD environment variables match the server configuration
- **No detections found**: Try adding more training images or adjusting the server's detection parameters