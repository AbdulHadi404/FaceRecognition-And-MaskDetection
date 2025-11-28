# Face Recognition & Mask Detection System

A standalone attendance management system that uses face recognition and mask detection for automated, contactless attendance tracking. Perfect for educational institutes, offices, or any facility requiring attendance monitoring.

## ✨ Features

- **Real-time Face Recognition**: Uses LBPH, FisherFace, and EigenFace algorithms
- **Mask Detection**: Custom HAAR-Cascade classifier for mask detection
- **Dual Camera Support**: Simultaneous entry and exit tracking
- **Automatic Attendance Logging**: Timestamp-based attendance records
- **CSV-based Storage**: No database required - all data stored in CSV files
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

### Step 1: Register a New Student

1. Launch `main_window.py`
2. Click **"01 Enter New Student"**
3. Enter the student's name when prompted
4. Position the student in front of the camera
5. The system will automatically capture 10 images

### Step 2: Train the Models

1. Click **"02 Train Model"**
2. The system will load all collected images
3. Three recognition models will be trained (EigenFace, FisherFace, LBPH)
4. Training completes automatically

### Step 3: Start Attendance Tracking

1. Click **"03 Face Attendance & Mask Detection"**
2. Two camera windows will open:
   - **Entry Cam**: Monitors entrance (with mask detection)
   - **Exit Cam**: Monitors exit
3. The system will automatically recognize faces and log attendance
4. Press **ESC** to stop tracking

### Step 4: Consolidate Attendance

1. Click **"04 Save Face Attendance File"**
2. The system merges entry and exit records
3. Calculates engagement time for each person
4. Saves consolidated report to `attendance_results/`

## 📁 Project Structure

```
FaceRecognition-And-MaskDetection/
├── main_window.py              # Main GUI application
├── collect_images.py           # Image collection module
├── train_models.py             # Model training module
├── face_recognition.py         # Face recognition & detection
├── consolidate_attendance.py   # Attendance consolidation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── HOW_IT_WORKS.md            # Detailed scenario guide
├── img.jpg                     # GUI background image (optional)
├── xml/                        # Haar Cascade classifiers
│   ├── frontal_face.xml        # Face detection
│   └── mask_cascade.xml        # Mask detection
├── members/                    # Student face images (created automatically)
├── attendance_in/              # Entry attendance records
├── attendance_out/             # Exit attendance records
└── attendance_results/         # Consolidated reports
```

## 🔧 Requirements

### Python Libraries

- `opencv-python` - Computer vision and image processing
- `opencv-contrib-python` - Extended OpenCV features (face recognition)
- `matplotlib` - Image display and visualization
- `numpy` - Numerical operations
- `pandas` - Data manipulation for attendance records
- `Pillow` - Image processing for GUI

### Built-in Libraries

- `tkinter` - GUI framework
- `json`, `time`, `datetime`, `sys`, `os`, `glob` - Standard library modules

## 📚 Documentation

- **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**: Detailed scenario-based explanation of how the system works
- **Code Documentation**: All Python files include comprehensive docstrings and comments

## 🎯 How It Works

### Image Collection (`collect_images.py`)

- Captures 10 face images per person
- Detects faces using Haar Cascade classifier
- Normalizes images (grayscale, histogram equalization, resizing)
- Saves images to `members/{name}/` directory

### Model Training (`train_models.py`)

- Loads all collected face images
- Trains three recognition algorithms:
  - **EigenFace**: PCA-based recognition
  - **FisherFace**: LDA-based recognition
  - **LBPH**: Local Binary Patterns (most robust)
- Creates label mappings for each person

### Face Recognition (`face_recognition.py`)

- Real-time face detection and recognition
- Mask detection using custom Haar Cascade
- Dual camera processing (entry and exit)
- Automatic attendance logging to CSV files
- Visual feedback with bounding boxes and labels

### Attendance Consolidation (`consolidate_attendance.py`)

- Merges entry and exit records by person name
- Calculates engagement time (time difference)
- Generates consolidated reports with all attendance data

## 🎨 Application Interface

The system features a modern, user-friendly GUI with:

- Dark theme color scheme
- Clear menu navigation
- Button hover effects
- Real-time status updates

## ⚙️ Configuration

### Camera Setup

- **Camera 0**: Entry camera (default webcam)
- **Camera 1**: Exit camera (USB camera)

You can modify camera indices in `face_recognition.py` if needed.

### Recognition Thresholds

- **Entry Threshold**: 76 (LBPH confidence)
- **Exit Threshold**: 75 (LBPH confidence)

Lower confidence values indicate better matches for LBPH algorithm.

## 📊 Output Format

### Entry Records (`attendance_in/`)

```csv
Name,Date,Time,Mask
john doe,2024-01-15,09:00:00,True
```

### Exit Records (`attendance_out/`)

```csv
Name,Date,Time
john doe,2024-01-15,17:00:00
```

### Consolidated Reports (`attendance_results/`)

```csv
Name,DateIn,TimeIn,Mask,DateOut,TimeOut,Engage-Min,Engage-Hrs
john doe,2024-01-15,09:00:00,True,2024-01-15,17:00:00,480,8.00
```

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

- Make sure `members/` directory exists
- Collect images first using `collect_images.py`
- Check that images are saved in correct format (.jpg)

## 🤝 Contributing

This is a standalone project, but suggestions and improvements are welcome!

## 📝 License

This project is available for educational and personal use.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- Haar Cascade classifiers for face and mask detection
- LBPH, FisherFace, and EigenFace algorithms for face recognition

---

**Note**: This system provides an automated, contactless attendance tracking solution perfect for educational institutes, offices, or any facility requiring attendance monitoring.

For detailed explanations and scenarios, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
