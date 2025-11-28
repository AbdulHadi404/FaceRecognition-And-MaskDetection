# Face Recognition & Mask Detection System

A standalone application for real-time face recognition and mask detection using computer vision. Perfect for access control, security monitoring, or any application requiring face identification and mask compliance detection.

## ✨ Features

- **Real-time Face Recognition**: Uses LBPH, FisherFace, and EigenFace algorithms
- **Mask Detection**: Custom HAAR-Cascade classifier for mask detection
- **Real-time Video Processing**: Live camera feed with face recognition and mask detection
- **Multiple Recognition Algorithms**: Three different algorithms for robust face identification
- **Standalone Application**: No external dependencies or cloud services
- **Modern GUI**: User-friendly interface built with Tkinter

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam(s) connected to your computer
- Windows, macOS, or Linux

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main_window.py
   ```

## 📖 Usage Guide

### Step 1: Register a New Person

1. Launch `main_window.py`
2. Click **"01 Enter New Person"**
3. Enter the person's name when prompted
4. Position the person in front of the camera
5. The system will automatically capture 10 images

### Step 2: Train the Models

1. Click **"02 Train Model"**
2. The system will load all collected images
3. Recognition models will be trained (EigenFace, FisherFace, LBPH)
4. Training completes automatically

### Step 3: Start Face Recognition & Mask Detection

1. Click **"03 Face Recognition & Mask Detection"**
2. A camera window will open showing live video feed
3. The system will automatically:
   - Detect faces in real-time
   - Recognize registered persons
   - Detect if masks are being worn
   - Display recognition results and mask status
4. Press **ESC** to stop

## 📁 Project Structure

```
FaceRecognition-And-MaskDetection/
├── main_window.py              # Main GUI application (entry point)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── collect_images.py       # Image collection module
│   ├── train_models.py         # Model training module
│   ├── face_recognition.py     # Face recognition & detection
│   └── gui_messages.py         # GUI message helpers
│
├── resources/                  # Static resources
│   ├── images/
│   │   └── img.jpg             # GUI background image (optional)
│   └── xml/                    # Haar Cascade classifiers
│       ├── frontal_face.xml    # Face detection
│       └── mask_cascade.xml    # Mask detection
│
├── docs/                       # Documentation
│   ├── HOW_IT_WORKS.md        # Detailed scenario guide
│   └── TECHNICAL_DOCUMENTATION.md # Technical details
│
├── build_config/               # Build configuration
│   └── face_recognition_app.spec # PyInstaller spec file
│
└── [Runtime directories - created automatically]
    └── members/                # Face images storage
```

## 🔧 Requirements

### Python Libraries

- `opencv-python` - Computer vision and image processing
- `opencv-contrib-python` - Extended OpenCV features (face recognition)
- `matplotlib` - Image display and visualization
- `numpy` - Numerical operations
- `Pillow` - Image processing for GUI

### Built-in Libraries

- `tkinter` - GUI framework
- `json`, `time`, `datetime`, `sys`, `os`, `glob` - Standard library modules

## 📚 Documentation

- **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**: Detailed scenario-based explanation of how the system works
- **[docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)**: Technical details about algorithms and implementation
- **Code Documentation**: All Python files include comprehensive docstrings and comments

## 🎯 How It Works

### Image Collection (`src/collect_images.py`)

- Captures 10 face images per person
- Detects faces using Haar Cascade classifier (`resources/xml/frontal_face.xml`)
- Normalizes images (grayscale, histogram equalization, resizing)
- Saves images to `members/{name}/` directory

### Model Training (`src/train_models.py`)

- Loads all collected face images
- Trains recognition algorithms:
  - **EigenFace**: PCA-based recognition (works with 1+ people)
  - **FisherFace**: LDA-based recognition (requires 2+ people, skipped if only 1 person)
  - **LBPH**: Local Binary Patterns (most robust, works with 1+ people)
- Creates label mappings for each person
- **Note**: FisherFace requires at least 2 people in the dataset. With only 1 person, EigenFace and LBPH will be trained successfully.

### Face Recognition (`src/face_recognition.py`)

- Real-time face detection and recognition
- Mask detection using custom Haar Cascade (`resources/xml/mask_cascade.xml`)
- Single camera support (default webcam)
- Visual feedback with bounding boxes and labels
- Uses LBPH algorithm for recognition (most robust)
- Displays person name and mask status in real-time

## 🎨 Application Interface

The system features a modern, user-friendly GUI with:

- Dark theme color scheme
- Clear menu navigation
- Button hover effects
- Real-time status updates

## ⚙️ Configuration

### Camera Setup

- **Camera 0**: Default webcam

You can modify camera indices in `src/face_recognition.py` if needed.

### Recognition Thresholds

- **LBPH Threshold**: 76 (confidence level)

Lower confidence values indicate better matches for LBPH algorithm. If confidence is above the threshold, the face is marked as "Unknown".

## 🛠️ Troubleshooting

### Camera Not Detected

- Ensure webcam is connected and not being used by another application
- Check camera permissions in your operating system
- Try different camera indices (0, 1, 2) in the code

### Face Not Recognized

- Ensure person is registered (images collected)
- Check lighting conditions
- Retrain models after adding new people
- Adjust recognition threshold if needed

### No Images Found

- Make sure `members/` directory exists (created automatically)
- Collect images first using "Enter New Person" button
- Check that images are saved in correct format (.jpg)

### Training Error: "At least two classes are needed for LDA"

- **FisherFace requires at least 2 people** in the dataset
- This is normal if you only have 1 person registered
- The system will automatically skip FisherFace training with 1 person
- EigenFace and LBPH will still train successfully
- Add a second person to enable FisherFace training

## 🤝 Contributing

This is a standalone project, but suggestions and improvements are welcome!

## 📝 License

This project is available for educational and personal use.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- Haar Cascade classifiers for face and mask detection
- LBPH, FisherFace, and EigenFace algorithms for face recognition

---

**Note**: This system provides real-time face recognition and mask detection capabilities, perfect for access control, security monitoring, or any application requiring face identification and mask compliance detection.

For detailed explanations and scenarios, see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).
