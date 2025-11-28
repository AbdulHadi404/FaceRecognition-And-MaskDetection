"""
Face Recognition & Mask Detection Module - Attendance System
==============================================================

This is the main module for real-time face recognition and mask detection.
It uses a single camera to track attendance automatically.

Features:
- Real-time face recognition using LBPH algorithm
- Mask detection using custom Haar Cascade
- Single camera support (default webcam)
- Automatic attendance logging to CSV files
- Visual feedback with bounding boxes and labels

Workflow:
1. Loads trained face recognition models
2. Opens default webcam (Camera 0)
3. Continuously detects faces and masks in camera feed
4. Recognizes faces and saves attendance records
5. Displays live video with annotations

Camera Setup:
- Camera 0: Default webcam (with mask detection)

Requirements:
- Trained models (run 'train_models.py' first)
- resources/xml/frontal_face.xml (face detection)
- resources/xml/mask_cascade.xml (mask detection)
- members/ directory with face images
"""

import cv2
import numpy as np
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
import time
import datetime

# Add project root to path for imports and resource access
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Also add src directory to path for module imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Change working directory to project root to ensure relative paths work
os.chdir(project_root)

from gui_messages import show_info, show_warning, show_error


# ============================================================================
# HELPER FUNCTIONS - RESOURCE PATH
# ============================================================================

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and PyInstaller.
    
    Args:
        relative_path: Path relative to the project root
        
    Returns:
        Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Running in development mode - use project root
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    return os.path.join(base_path, relative_path)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Recognition threshold (LBPH: lower is better, but we use > threshold for unknown)
# Typical values: EigenFace ~3375, FisherFace ~1175, LBPH ~65-76
LBPH_THRESHOLD = 76  # Threshold for face recognition

# Mask detection parameters
MASK_SCALE_FACTOR = 1.2
MASK_MIN_NEIGHBORS = 5
MASK_MIN_SIZE = (100, 100)
MASK_MAX_SIZE = (150, 150)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def plt_show(image, title=""):
    """
    Display an image using matplotlib (for debugging).
    
    Args:
        image: Image array
        title: Title for the image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()


# ============================================================================
# CLASSES
# ============================================================================

class FaceDetector:
    """
    Detects faces in images using Haar Cascade classifier.
    """
    
    def __init__(self, xml_path):
        """
        Initialize face detector.
        
        Args:
            xml_path: Path to Haar Cascade XML file
        """
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        """
        Detect faces in an image.
        
        Args:
            image: Input image
            biggest_only: If True, detect only the largest face
            
        Returns:
            List of face coordinates (x, y, width, height)
        """
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (75, 75)
        
        if biggest_only:
            flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
        else:
            flags = cv2.CASCADE_SCALE_IMAGE
        
        faces_coord = self.classifier.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags
        )
        return faces_coord


class VideoCamera:
    """
    Handles video capture from webcam.
    """
    
    def __init__(self, index=0):
        """
        Initialize video camera.
        
        Args:
            index: Camera index (0 = Default webcam)
        """
        self.video = cv2.VideoCapture(index)
        self.index = index
        print(f"Camera {index} opened: {self.video.isOpened()}")
    
    def __del__(self):
        """Release camera resources."""
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        """
        Capture a frame from the camera.
        
        Args:
            in_grayscale: If True, convert to grayscale
            
        Returns:
            Frame as numpy array
        """
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def cut_faces(image, faces_coord):
    """
    Crop face regions from image.
    
    Args:
        image: Input image
        faces_coord: List of face coordinates
        
    Returns:
        List of cropped face images
    """
    faces = []
    for (x, y, w, h) in faces_coord:
        # Remove 30% from width edges to focus on face center
        w_rm = int(0.3 * w / 2)
        face = image[y: y + h, x + w_rm: x + w - w_rm]
        faces.append(face)
    return faces


def normalize_intensity(images):
    """
    Normalize image intensity using histogram equalization.
    
    Args:
        images: List of images
        
    Returns:
        List of normalized grayscale images
    """
    images_norm = []
    for image in images:
        if len(image.shape) == 3:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = cv2.equalizeHist(image)
        images_norm.append(normalized)
    return images_norm


def resize(images, size=(100, 100)):
    """
    Resize images to a standard size.
    
    Args:
        images: List of images
        size: Target size (width, height)
        
    Returns:
        List of resized images
    """
    images_resized = []
    for image in images:
        if image.shape[0] < size[0] or image.shape[1] < size[1]:
            resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_resized.append(resized)
    return images_resized


def normalize_faces(frame, faces_coord):
    """
    Complete face normalization pipeline.
    
    Args:
        frame: Input frame
        faces_coord: Face coordinates
        
    Returns:
        List of normalized face images
    """
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces


def draw_rectangle(image, coords):
    """
    Draw rectangles around detected faces.
    
    Args:
        image: Image to draw on
        coords: List of face coordinates
    """
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2)
        # Green rectangle for recognized faces
        cv2.rectangle(
            image, 
            (x + w_rm, y), 
            (x + w - w_rm, y + h), 
            (102, 255, 0),  # Green color (BGR)
            1  # Thickness
        )


# ============================================================================
# DATASET LOADING
# ============================================================================

def collect_dataset():
    """
    Load all face images from the members/ directory.
    
    Returns:
        images: List of face images
        labels: Array of numeric labels
        labels_dic: Dictionary mapping labels to person names
    """
    images = []
    labels = []
    labels_dic = {}
    
    members_dir = "members/"
    if not os.path.exists(members_dir):
        print(f"Error: Directory '{members_dir}' not found!")
        return None, None, None
    
    members = [person for person in os.listdir(members_dir) 
               if os.path.isdir(os.path.join(members_dir, person))]
    
    if len(members) == 0:
        print(f"Error: No person directories found in '{members_dir}'")
        return None, None, None
    
    # Load images for each person
    for i, person in enumerate(members):
        labels_dic[i] = person
        person_dir = os.path.join(members_dir, person)
        
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_dir, image_file)
                img = cv2.imread(image_path, 0)  # Read as grayscale
                if img is not None:
                    images.append(img)
                    labels.append(i)
    
    print(f"\nLoaded {len(images)} images for {len(members)} person(s)")
    return images, np.array(labels), labels_dic


# ============================================================================
# INITIALIZATION
# ============================================================================

# Load dataset and train models
images, labels, labels_dic = collect_dataset()

if images is None or len(images) == 0:
    show_error(
        "No Training Data",
        "No training data found!\n\n"
        "Please:\n"
        "1. Collect images (Enter New Student)\n"
        "2. Train models (Train Model)\n\n"
        "Then try again."
    )
    sys.exit(1)

# Train LBPH model (used for recognition)
rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)

# Initialize detectors
try:
    detector = FaceDetector(resource_path("resources/xml/frontal_face.xml"))  # Face detector
    detector_mask = cv2.CascadeClassifier(resource_path("resources/xml/mask_cascade.xml"))  # Mask detector
except Exception as e:
    show_error("Initialization Error", f"Failed to load detectors:\n{str(e)}")
    sys.exit(1)

# Initialize camera
try:
    camera = VideoCamera(0)  # Default webcam
    if not camera.video.isOpened():
        show_error("Camera Error", "Could not open camera.\nPlease check your camera connection.")
        sys.exit(1)
except Exception as e:
    show_error("Camera Error", f"Failed to initialize camera:\n{str(e)}")
    sys.exit(1)

# Get current date and time
current_time = time.time()
date = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d')
timeStamp = datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')


# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================

# Create output directory if it doesn't exist
os.makedirs("attendance_in", exist_ok=True)

while True:
    # ========================================================================
    # CAMERA PROCESSING (with mask detection)
    # ========================================================================
    
    frame = camera.get_frame()
    mask_detected = False
    
    # Detect masks
    masks = detector_mask.detectMultiScale(
        frame,
        scaleFactor=MASK_SCALE_FACTOR,
        minNeighbors=MASK_MIN_NEIGHBORS,
        minSize=MASK_MIN_SIZE,
        maxSize=MASK_MAX_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw mask detection results
    for (x, y, w, h) in masks:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
        cv2.putText(
            frame, 
            'Using Mask',
            (x, y + h + 30), 
            cv2.FONT_HERSHEY_PLAIN, 
            1.5, 
            (255, 255, 255), 
            2
        )
        mask_detected = True
    
    # Detect faces
    faces_coord = detector.detect(frame, biggest_only=False)  # Detect multiple faces
    attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Mask'])
    
    if len(faces_coord) > 0:
        # Normalize detected faces
        faces = normalize_faces(frame, faces_coord)
        
        # Recognize each face
        for i, face in enumerate(faces):
            collector = cv2.face.StandardCollector_create()
            rec_lbph.predict_collect(face, collector)
            confidence = collector.getMinDist()
            predicted_label = collector.getMinLabel()
            
            person_name = labels_dic[predicted_label]
            print(f"\nPerson: {person_name.capitalize()}, "
                  f"Confidence: {round(confidence)}, Mask: {mask_detected}")
            
            # Check if confidence is below threshold (good match)
            if confidence > LBPH_THRESHOLD:
                # Unknown person (confidence too high)
                cv2.putText(
                    frame,
                    "Unknown",
                    (faces_coord[i][0], faces_coord[i][1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (66, 55, 245),  # Red color
                    1
                )
            else:
                # Recognized person - save attendance
                cv2.putText(
                    frame,
                    person_name.capitalize(),
                    (faces_coord[i][0], faces_coord[i][1] - 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (102, 255, 0),  # Green color
                    1
                )
                
                # Save attendance record
                attendance_df.loc[len(attendance_df)] = [
                    person_name,
                    date,
                    timeStamp,
                    str(mask_detected)
                ]
                
                # Create filename with timestamp
                hour, minute, second = timeStamp.split(":")
                filename = (f"attendance_in/Attendance_{person_name}-{date}_"
                          f"{hour}-{minute}-{second}.csv")
                attendance_df.to_csv(filename, index=False)
        
        # Draw rectangles around faces
        draw_rectangle(frame, faces_coord)
    
    # Add exit instruction
    cv2.putText(
        frame,
        "ESC to exit",
        (5, frame.shape[0] - 5),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    
    # Display camera feed
    cv2.imshow("Face Recognition - Attendance", frame)
    
    # ========================================================================
    # EXIT CONDITION
    # ========================================================================
    
    # Check for ESC key press (ASCII code 27)
    key = cv2.waitKey(33) & 0xFF
    if key == 27:  # ESC key
        print("\nExiting...")
        cv2.destroyAllWindows()
        break

# Cleanup
del camera
