# Logging Documentation

## Overview

The application includes comprehensive file-based logging to help debug issues, especially when running as an executable.

## Log File Locations

### Development Mode
Logs are written to: `logs/` directory in the project root
- Example: `logs/face_recognition_20241128.log`

### Executable Mode
Logs are written to: `logs/` directory next to the executable
- Example: If executable is at `C:\Programs\FaceRecognitionApp.exe`, logs will be at `C:\Programs\logs\`

### Fallback Location
If logs cannot be written to the above locations, they will be written to:
- `%TEMP%\FaceRecognitionApp_logs\` (Windows temp directory)

## Log Files

Each module creates its own log file:
- `main_window_YYYYMMDD.log` - Main GUI application
- `collect_images_YYYYMMDD.log` - Image collection module
- `train_models_YYYYMMDD.log` - Model training module
- `face_recognition_YYYYMMDD.log` - Face recognition module

## Log Levels

- **DEBUG**: Detailed diagnostic information (paths, variable values, etc.)
- **INFO**: General informational messages (startup, completion, etc.)
- **WARNING**: Warning messages (non-critical issues)
- **ERROR**: Error messages (failures, exceptions)
- **EXCEPTION**: Full exception traces with stack traces

## What's Logged

### Initialization
- Module startup
- Project root path
- Current working directory
- Executable vs development mode
- Python version
- Log file location

### Resource Loading
- XML file paths (face detection, mask detection)
- Image paths
- Whether files exist

### Camera Operations
- Camera initialization attempts
- Camera index used
- Camera properties (width, height)
- Frame capture success/failure
- Window creation

### Model Training
- Dataset loading progress
- Number of images and people loaded
- Model training progress (EigenFace, FisherFace, LBPH)
- Training success/failure

### Face Recognition
- Face detection results
- Recognition results with confidence scores
- Mask detection results
- Errors during recognition

### Errors
- Full exception traces
- Error context (what operation failed)
- Path information when file errors occur

## Finding Log Files

When debugging executable issues:

1. **Check next to the executable**: Look for a `logs/` folder in the same directory as `FaceRecognitionApp.exe`

2. **Check temp directory**: 
   - Windows: `%TEMP%\FaceRecognitionApp_logs\`
   - Or run: `echo %TEMP%` to find your temp directory

3. **Check console output**: When the executable starts, it prints the log file location to console

## Example Log Entry

```
2024-11-28 14:05:53 - face_recognition - INFO - face_recognition.py:460 - Starting main detection loop...
2024-11-28 14:05:54 - face_recognition - DEBUG - face_recognition.py:468 - resource_path: 'resources/xml/frontal_face.xml' -> 'C:\...\resources\xml\frontal_face.xml'
2024-11-28 14:05:54 - face_recognition - ERROR - face_recognition.py:442 - Failed to initialize camera: [Error message]
2024-11-28 14:05:54 - face_recognition - ERROR - face_recognition.py:443 - Traceback (most recent call last):
  File "src/face_recognition.py", line 429, in recognize
    camera = VideoCamera(0)
  ...
```

## Debugging Tips

1. **Check the log file location**: The first message in each log shows where it's being written
2. **Look for ERROR/EXCEPTION entries**: These show what failed and why
3. **Check DEBUG entries**: These show path resolutions and resource loading
4. **Verify file paths**: Logs show the actual paths being used, which helps identify path issues
5. **Check camera initialization**: Look for "Camera initialized successfully" or error messages

## Disabling Logging

To disable file logging (console only), modify `src/logger_setup.py` and comment out the file handler creation.

