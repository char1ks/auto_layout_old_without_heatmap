#!/usr/bin/env python3
"""
Run script for the ML Segmentation Gradio Client

This script launches the Gradio web interface for interacting with the
ML segmentation API server.
"""

import os
import sys
from app import create_interface, client

def main():
    # Check if server is accessible
    print("Checking server connection...")
    if not client.health_check():
        print("⚠️  Warning: Cannot connect to the API server")
        print(f"   Make sure the server is running at {client.base_url}")
        print("   You can still start the client, but it won't work until the server is available")
        print()
    else:
        print("✅ Server is accessible")
        print()
    
    print("Starting Gradio client...")
    print("The web interface will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop")
    print()
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()