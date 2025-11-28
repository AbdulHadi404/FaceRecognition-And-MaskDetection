# IP Camera Setup Guide for Android Smartphone

This guide will help you configure your Android smartphone as an IP camera to use with the Face Recognition & Mask Detection application.

## Recommended Android Apps

### Option 1: IP Webcam (Free) - Recommended
- **Download**: [IP Webcam on Google Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam)
- **Features**: HTTP/MJPEG streaming, RTSP support, no watermark

### Option 2: DroidCam (Free/Paid)
- **Download**: [DroidCam on Google Play Store](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
- **Features**: WiFi and USB modes, good quality

### Option 3: AtHome Camera (Free)
- **Download**: [AtHome Camera on Google Play Store](https://play.google.com/store/apps/details?id=com.veracy.android.camera)
- **Features**: Easy setup, multiple stream formats

## Setup Instructions for IP Webcam (Recommended)

### Step 1: Install IP Webcam on Your Android Phone

1. Install "IP Webcam" from the Google Play Store
2. Open the app

### Step 2: Configure IP Webcam

1. **Start the server**:
   - Scroll down and tap "Start server"
   - Note the IP address shown (e.g., `192.168.1.100:8080`)

2. **Video preferences** (optional):
   - Tap "Video preferences"
   - Set resolution (e.g., 640x480 or 1280x720)
   - Set FPS (15-30 recommended)

3. **Video codec**:
   - Use "MJPEG" or "H.264" (MJPEG recommended for better compatibility)

### Step 3: Get the Stream URL

The app will display several URLs. You need the **MJPEG stream URL**:

- **MJPEG Stream**: `http://YOUR_PHONE_IP:8080/video`
  - Example: `http://192.168.1.100:8080/video`
  
- **Alternative formats**:
  - RTSP: `rtsp://YOUR_PHONE_IP:8086/h264_pcm.sdp`
  - HTTP (if RTSP doesn't work): `http://YOUR_PHONE_IP:8080/video`

### Step 4: Configure the Application

#### Method 1: Using Configuration File (Recommended)

1. Create/edit the file: `config/camera_settings.json`
2. Add your IP camera URL:

```json
{
    "source": "http://192.168.1.100:8080/video"
}
```

Replace `192.168.1.100` with your phone's IP address.

#### Method 2: Using Python Script

Run this Python script to set the camera source:

```python
from config.camera_config import save_camera_config

# Set IP camera URL
save_camera_config("http://192.168.1.100:8080/video")

# Or set back to local webcam
# save_camera_config(0)
```

### Step 5: Ensure Both Devices Are on Same Network

- Your computer and Android phone must be connected to the **same WiFi network**
- Check your phone's IP address in the IP Webcam app

## Testing the Connection

### Test with VLC Media Player (Optional)

1. Open VLC Media Player
2. Go to Media → Open Network Stream
3. Enter your stream URL: `http://YOUR_PHONE_IP:8080/video`
4. Click Play to verify the stream works

### Test with the Application

1. Make sure your IP camera URL is configured in `config/camera_settings.json`
2. Run the main application:
   ```bash
   python main_window.py
   ```
3. Try "Enter New Person" or "Face Recognition & Mask Detection"
4. The video feed should appear from your phone

## Troubleshooting

### Connection Issues

**Problem**: Camera fails to open
- **Solution**: 
  - Verify both devices are on the same WiFi network
  - Check firewall settings (allow port 8080)
  - Verify the IP address is correct
  - Try accessing the URL in a web browser first

**Problem**: High latency or lag
- **Solution**:
  - Reduce video resolution in IP Webcam settings
  - Lower FPS (15-20 recommended)
  - Ensure stable WiFi connection
  - Try using RTSP stream instead of HTTP

**Problem**: Stream disconnects frequently
- **Solution**:
  - Keep phone plugged into charger
  - Disable battery optimization for IP Webcam app
  - Use 5GHz WiFi if available (better performance)

### Finding Your Phone's IP Address

1. In IP Webcam app, the IP is displayed when you start the server
2. Or check: Settings → WiFi → Tap on your network → View IP address
3. Or use: Settings → About Phone → Status → IP Address

### URL Format Examples

- **HTTP/MJPEG**: `http://192.168.1.100:8080/video`
- **RTSP**: `rtsp://192.168.1.100:8086/h264_pcm.sdp`
- **With credentials** (if set): `http://username:password@192.168.1.100:8080/video`

### Switching Back to Local Webcam

To switch back to your computer's webcam, set the source to `0`:

```json
{
    "source": 0
}
```

Or delete the `config/camera_settings.json` file to use default.

## Security Notes

- IP Webcam streams are only accessible on your local network
- For security, you can set a username/password in IP Webcam settings
- Never expose this to the internet without proper security measures

## Performance Tips

1. **Resolution**: Lower resolution = better performance
   - Recommended: 640x480 or 800x600 for face recognition
   
2. **Frame Rate**: 15-20 FPS is usually sufficient
   
3. **Codec**: MJPEG generally works better than H.264 with OpenCV

4. **Network**: Use 5GHz WiFi for better performance if available

## Alternative: USB Connection (DroidCam)

If WiFi is unreliable, DroidCam supports USB connection:

1. Install DroidCam on phone and DroidCam Client on computer
2. Connect via USB
3. Use the virtual camera device (usually camera index 1 or 2)
4. Configure in `camera_settings.json`:
   ```json
   {
       "source": 1
   }
   ```

---

**Need Help?**
- Check the IP Webcam app's help section
- Verify network connectivity with `ping YOUR_PHONE_IP`
- Test stream URL in a web browser first

