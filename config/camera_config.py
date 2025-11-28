"""
Camera Configuration
====================

Configuration for camera sources (local webcam or IP camera).

Usage:
    - For local webcam: Set CAMERA_SOURCE to a number (e.g., 0, 1)
    - For IP camera: Set CAMERA_SOURCE to the camera URL
    
IP Camera URL formats:
    - RTSP: rtsp://username:password@ip:port/stream
    - HTTP/MJPEG: http://ip:port/video or http://ip:port/video.mjpg
    - For Android IP camera apps, check the app's settings for the stream URL
"""

import os
import json

# Default configuration file path
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'camera_settings.json')

# Default camera source (0 = default webcam)
# Set to an integer for local camera index, or a URL string for IP camera
CAMERA_SOURCE = 0  # Can be: 0, 1, 2, etc. for local cameras, or URL like "rtsp://..." or "http://..."

def load_camera_config():
    """
    Load camera configuration from JSON file if it exists.
    
    Returns:
        dict: Camera configuration with 'source' key
    """
    config = {'source': CAMERA_SOURCE}
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                file_config = json.load(f)
                if 'source' in file_config:
                    config['source'] = file_config['source']
        except Exception as e:
            print(f"Warning: Could not load camera config: {e}")
            print(f"Using default camera source: {CAMERA_SOURCE}")
    
    return config

def save_camera_config(camera_source):
    """
    Save camera configuration to JSON file.
    
    Args:
        camera_source: Camera source (integer index or URL string)
    """
    config_dir = os.path.dirname(CONFIG_FILE)
    os.makedirs(config_dir, exist_ok=True)
    
    config = {'source': camera_source}
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Camera configuration saved: {camera_source}")
    except Exception as e:
        print(f"Warning: Could not save camera config: {e}")

def get_camera_source():
    """
    Get the current camera source configuration.
    
    Returns:
        Camera source (integer index or URL string)
    """
    config = load_camera_config()
    return config['source']

