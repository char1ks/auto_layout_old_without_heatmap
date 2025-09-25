#!/usr/bin/env python3
"""
Test script for the ML Segmentation Gradio Client

This script tests the client functionality programmatically without the web interface.
"""

import os
import sys
from PIL import Image
import numpy as np
from app import APIClient, image_to_bytes

def create_test_image(size=(256, 256), color=(255, 0, 0)):
    """Create a simple test image"""
    return Image.new('RGB', size, color=color)

def test_client():
    # Create API client
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    username = os.getenv("APP_USERNAME", "admin")
    password = os.getenv("APP_PASSWORD", "secret")
    
    client = APIClient(f"http://{host}:{port}", username, password)
    
    print("Testing ML Segmentation Client")
    print("=" * 40)
    
    # Test 1: Health check
    print("1. Testing server connection...")
    if client.health_check():
        print("✅ Server is accessible")
    else:
        print("❌ Server is not accessible")
        print("Make sure the API server is running before running this test")
        return False
    
    # Test 2: Set training data
    print("\n2. Testing training data upload...")
    
    # Create some test training images
    red_image = create_test_image((256, 256), (255, 0, 0))  # Red for class "red"
    blue_image = create_test_image((256, 256), (0, 0, 255))  # Blue for class "blue"
    
    positive_images = {
        "red": [image_to_bytes(red_image)],
        "blue": [image_to_bytes(blue_image)]
    }
    
    if client.set_references(positive_images):
        print("✅ Training data sent successfully")
    else:
        print("❌ Failed to send training data")
        return False
    
    # Test 3: Inference
    print("\n3. Testing inference...")
    
    # Create a test inference image (mix of red and blue)
    inference_img = create_test_image((256, 256), (128, 0, 128))  # Purple
    
    results = client.infer([image_to_bytes(inference_img)])
    
    if results:
        print(f"✅ Inference successful, got {len(results)} result(s)")
        
        # Print detailed results
        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            masks = result.get("masks", [])
            print(f"  - Found {len(masks)} detection(s)")
            
            for j, mask in enumerate(masks):
                class_id = mask.get("class_id", mask.get("category_id", "unknown"))
                score = mask.get("score", mask.get("confidence", "N/A"))
                bbox = mask.get("bbox", [])
                polygons = mask.get("polygons", [])
                
                print(f"    Detection {j + 1}:")
                print(f"      Class: {class_id}")
                print(f"      Score: {score}")
                print(f"      Bbox: {bbox}")
                print(f"      Polygons: {len(polygons)} polygon(s)")
    else:
        print("❌ Inference failed or returned no results")
        return False
    
    print("\n" + "=" * 40)
    print("✅ All tests passed! The client is working correctly.")
    print("\nYou can now run the Gradio interface:")
    print("  python run.py")
    print("\nOr start the server directly:")
    print("  python app.py")
    
    return True

if __name__ == "__main__":
    success = test_client()
    sys.exit(0 if success else 1)