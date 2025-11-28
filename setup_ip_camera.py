"""
IP Camera Setup Helper Script
==============================

This script helps you configure an IP camera URL for use with the
Face Recognition & Mask Detection application.

Usage:
    python setup_ip_camera.py
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.camera_config import CONFIG_FILE, save_camera_config, get_camera_source

def main():
    print("=" * 60)
    print("IP Camera Configuration Setup")
    print("=" * 60)
    print()
    
    # Show current configuration
    current_source = get_camera_source()
    if isinstance(current_source, str):
        print(f"Current camera source: {current_source}")
    else:
        print(f"Current camera source: Local webcam (index {current_source})")
    print()
    
    print("Select camera source:")
    print("1. Local webcam (default)")
    print("2. IP camera (HTTP/MJPEG stream)")
    print("3. IP camera (RTSP stream)")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        # Local webcam
        camera_index = input("Enter camera index (0 for default, 1 for second camera, etc.): ").strip()
        try:
            index = int(camera_index)
            save_camera_config(index)
            print(f"\n✓ Configuration saved: Local webcam (index {index})")
        except ValueError:
            print("Invalid input. Please enter a number.")
            return
    
    elif choice == "2":
        # HTTP/MJPEG stream
        print("\nEnter IP camera URL (HTTP/MJPEG format)")
        print("Example: http://192.168.1.100:8080/video")
        print()
        ip = input("Enter your phone's IP address: ").strip()
        port = input("Enter port (default 8080): ").strip() or "8080"
        path = input("Enter video path (default /video): ").strip() or "/video"
        
        url = f"http://{ip}:{port}{path}"
        save_camera_config(url)
        print(f"\n✓ Configuration saved: {url}")
        print("\nMake sure:")
        print("1. Your phone and computer are on the same WiFi network")
        print("2. The IP camera app is running and streaming")
        print("3. Test the URL in a web browser first")
    
    elif choice == "3":
        # RTSP stream
        print("\nEnter IP camera URL (RTSP format)")
        print("Example: rtsp://192.168.1.100:8086/h264_pcm.sdp")
        print()
        url = input("Enter RTSP URL: ").strip()
        if url:
            save_camera_config(url)
            print(f"\n✓ Configuration saved: {url}")
        else:
            print("Invalid URL.")
            return
    
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please select 1-4.")
        return
    
    print("\n" + "=" * 60)
    print("Configuration complete!")
    print("=" * 60)
    print("\nYou can now run the application:")
    print("  python main_window.py")
    print("\nTo change the camera source later, run this script again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

