"""
Model Training Module - Face Recognition & Mask Detection System
================================================================

This module trains face recognition models using collected face images.
It uses three different algorithms: EigenFace, FisherFace, and LBPH.

Algorithms:
- EigenFace: Principal Component Analysis (PCA) based
- FisherFace: Linear Discriminant Analysis (LDA) based
- LBPH: Local Binary Patterns Histograms (most robust)

The trained models are used for real-time face recognition in the detection module.

Requirements:
- members/ directory with subdirectories containing face images
"""

import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

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

# Setup logging
from logger_setup import setup_logger
logger = setup_logger('train_models')

logger.info("="*60)
logger.info("Model Training Module Started")
logger.info("="*60)
logger.debug(f"Project root: {project_root}")
logger.debug(f"Current working directory: {os.getcwd()}")

from gui_messages import show_info, show_warning, show_error, show_progress


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def plt_show(image, title=""):
    """
    Display an image using matplotlib.
    
    Args:
        image: Image array (BGR or grayscale)
        title: Title for the displayed image
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
            index: Camera index (0 for default webcam)
        """
        self.video = cv2.VideoCapture(index)
        self.index = index
        print(f"Camera opened: {self.video.isOpened()}")
    
    def __del__(self):
        """Release camera resources."""
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        """
        Capture a frame from the camera.
        
        Args:
            in_grayscale: If True, convert frame to grayscale
            
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
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
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
        cv2.rectangle(
            image, 
            (x + w_rm, y), 
            (x + w - w_rm, y + h), 
            (200, 200, 0),  # Yellow color
            4
        )


# ============================================================================
# DATASET COLLECTION
# ============================================================================

def collect_dataset():
    """
    Load all face images from the members/ directory.
    
    Returns:
        images: List of face images (as numpy arrays)
        labels: Array of numeric labels for each image
        labels_dic: Dictionary mapping label numbers to person names
    """
    images = []
    labels = []
    labels_dic = {}
    
    # Get list of all person directories
    members_dir = "members/"
    if not os.path.exists(members_dir):
        print(f"Error: Directory '{members_dir}' not found!")
        print("Please collect images first using 'collect_images.py'")
        return None, None, None
    
    members = [person for person in os.listdir(members_dir) 
               if os.path.isdir(os.path.join(members_dir, person))]
    
    if len(members) == 0:
        print(f"Error: No person directories found in '{members_dir}'")
        print("Please collect images first using 'collect_images.py'")
        return None, None, None
    
    # Load images for each person
    for i, person in enumerate(members):
        labels_dic[i] = person  # Map label number to person name
        person_dir = os.path.join(members_dir, person)
        
        # Load all images for this person
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_dir, image_file)
                # Read image as grayscale (0 = grayscale mode)
                img = cv2.imread(image_path, 0)
                if img is not None:
                    images.append(img)
                    labels.append(i)  # Assign label number
    
    print(f"\nLoaded {len(images)} images for {len(members)} person(s):")
    for label, name in labels_dic.items():
        count = labels.count(label)
        print(f"  - {name}: {count} images")
    
    return images, np.array(labels), labels_dic


# ============================================================================
# MODEL TRAINING
# ============================================================================

# Collect progress messages
progress_messages = []
progress_messages.append("=" * 60)
progress_messages.append("FACE RECOGNITION MODEL TRAINING")
progress_messages.append("=" * 60)

# Load dataset
logger.info("Loading dataset...")
progress_messages.append("\nLoading dataset...")
images, labels, labels_dic = collect_dataset()

if images is None:
    error_msg = "Could not load training data"
    logger.error(error_msg)
    logger.error("collect_dataset() returned None")
    show_error(
        "Training Failed",
        "Could not load training data.\n\n"
        "Please collect images first using 'Enter New Person'."
    )
    sys.exit(1)

logger.info(f"Loaded {len(images)} images for {len(set(labels))} person(s)")

# Check number of unique people
num_people = len(set(labels))
progress_messages.append(f"\nFound {num_people} person(s) in dataset...")

# Train face recognition models
logger.info("Starting model training...")
progress_messages.append("\nTraining models...")

# 1. EigenFace Recognizer (PCA-based) - works with 1 or more people
logger.info("Training EigenFace model...")
progress_messages.append("  - Training EigenFace model...")
try:
    rec_eig = cv2.face.EigenFaceRecognizer_create()
    rec_eig.train(images, labels)
    logger.info("EigenFace model trained successfully")
    progress_messages.append("    ✓ EigenFace trained")
except Exception as e:
    error_msg = f"Failed to train EigenFace model: {e}"
    logger.exception(error_msg)
    progress_messages.append(f"    ✗ EigenFace training failed: {e}")
    raise

# 2. FisherFace Recognizer (LDA-based) - requires at least 2 people
if num_people >= 2:
    logger.info("Training FisherFace model...")
    progress_messages.append("  - Training FisherFace model...")
    try:
        rec_fisher = cv2.face.FisherFaceRecognizer_create()
        rec_fisher.train(images, labels)
        logger.info("FisherFace model trained successfully")
        progress_messages.append("    ✓ FisherFace trained")
    except Exception as e:
        error_msg = f"Failed to train FisherFace model: {e}"
        logger.exception(error_msg)
        progress_messages.append(f"    ✗ FisherFace training failed: {e}")
        raise
else:
    logger.info(f"Skipping FisherFace (requires at least 2 people, found {num_people})")
    progress_messages.append("  - Skipping FisherFace (requires at least 2 people)")
    progress_messages.append("    ℹ FisherFace will be skipped in recognition")

# 3. LBPH Recognizer (Local Binary Patterns - most robust) - works with 1 or more people
logger.info("Training LBPH model...")
progress_messages.append("  - Training LBPH model...")
try:
    rec_lbph = cv2.face.LBPHFaceRecognizer_create()
    rec_lbph.train(images, labels)
    logger.info("LBPH model trained successfully")
    progress_messages.append("    ✓ LBPH trained")
except Exception as e:
    error_msg = f"Failed to train LBPH model: {e}"
    logger.exception(error_msg)
    progress_messages.append(f"    ✗ LBPH training failed: {e}")
    raise

progress_messages.append("\n✓ All applicable models trained successfully!")
progress_messages.append("\nNote: Models are trained in memory. They will be used in the detection module.")


# ============================================================================
# SHOW RESULTS
# ============================================================================

# Show progress window
show_progress("Model Training Progress", progress_messages)
