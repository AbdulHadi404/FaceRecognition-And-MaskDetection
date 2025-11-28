"""
Image Collection Module - Face Recognition & Mask Detection System
==================================================================

This module captures face images from a webcam for training the face recognition system.
It collects 10 images per person and saves them in a normalized format.

Workflow:
1. User enters a name
2. Webcam captures 10 face images
3. Images are detected, cropped, normalized, and saved
4. Images are stored in members/{name}/ directory

Requirements:
- Webcam connected (camera index 0)
- resources/xml/frontal_face.xml (Haar Cascade classifier for face detection)
"""

import cv2
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
# HELPER FUNCTIONS
# ============================================================================

def plt_show(image, title=""):
    """
    Display an image using matplotlib.
    
    Args:
        image: Image array (BGR or grayscale)
        title: Title for the displayed image
    """
    # Convert BGR to RGB for matplotlib display
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()


# ============================================================================
# CLASSES
# ============================================================================

class VideoCamera:
    """
    Handles video capture from webcam.
    
    Attributes:
        video: VideoCapture object
        index: Camera index
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
        """Release camera resources when object is destroyed."""
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


class FaceDetector:
    """
    Detects faces in images using Haar Cascade classifier.
    
    Attributes:
        classifier: CascadeClassifier object
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
        # Detection parameters
        scale_factor = 1.2      # How much the image size is reduced at each scale
        min_neighbors = 5       # Minimum neighbors required for detection
        min_size = (75, 75)     # Minimum face size
        
        # Set detection flags
        if biggest_only:
            flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
        else:
            flags = cv2.CASCADE_SCALE_IMAGE
        
        # Detect faces
        faces_coord = self.classifier.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags
        )
        return faces_coord


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def cut_faces(image, faces_coord):
    """
    Crop face regions from image.
    
    Args:
        image: Input image
        faces_coord: List of face coordinates (x, y, width, height)
        
    Returns:
        List of cropped face images
    """
    faces = []
    for (x, y, w, h) in faces_coord:
        # Remove 20% from width edges to focus on face center
        w_rm = int(0.2 * w / 2)
        face = image[y: y + h, x + w_rm: x + w - w_rm]
        faces.append(face)
    return faces


def normalize_intensity(images):
    """
    Normalize image intensity using histogram equalization.
    Converts color images to grayscale and applies histogram equalization.
    
    Args:
        images: List of images
        
    Returns:
        List of normalized grayscale images
    """
    images_norm = []
    for image in images:
        # Convert to grayscale if color image
        if len(image.shape) == 3:  # Color image (BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better contrast
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
        # Use different interpolation based on whether upscaling or downscaling
        if image.shape[0] < size[0] or image.shape[1] < size[1]:
            # Upscaling: use INTER_AREA (better for enlarging)
            resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            # Downscaling: use INTER_CUBIC (better for shrinking)
            resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_resized.append(resized)
    return images_resized


def normalize_faces(frame, faces_coord):
    """
    Complete face normalization pipeline: crop, normalize intensity, resize.
    
    Args:
        frame: Input frame
        faces_coord: Face coordinates
        
    Returns:
        List of normalized face images
    """
    faces = cut_faces(frame, faces_coord)      # Step 1: Crop faces
    faces = normalize_intensity(faces)         # Step 2: Normalize intensity
    faces = resize(faces)                       # Step 3: Resize to standard size
    return faces


def draw_rectangle(image, coords):
    """
    Draw rectangles around detected faces.
    
    Args:
        image: Image to draw on
        coords: List of face coordinates (x, y, width, height)
    """
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2)  # Adjust width to match crop
        cv2.rectangle(
            image, 
            (x + w_rm, y), 
            (x + w - w_rm, y + h), 
            (200, 200, 0),  # Yellow color (BGR)
            4  # Thickness
        )


# ============================================================================
# MAIN IMAGE COLLECTION PROCESS
# ============================================================================

# Get student name from command line argument or prompt
if len(sys.argv) > 1:
    student_name = sys.argv[1].strip()
else:
    student_name = input('Enter student name: ').strip()

if not student_name:
    print("Error: No student name provided.")
    sys.exit(1)

student_name = student_name.lower()
folder = f"members/{student_name}"

print(f"\n{'='*60}")
print(f"IMAGE COLLECTION FOR: {student_name.upper()}")
print(f"{'='*60}\n")

# Initialize camera and face detector
print("Initializing camera...")
try:
    cv2.startWindowThread()  # Start window thread for better performance
    webcam = VideoCamera(0)  # Use camera index 0 (default webcam)
    if not webcam.video.isOpened():
        print("ERROR: Could not open camera. Please check your camera connection.")
        sys.exit(1)
    print("✓ Camera initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize camera: {e}")
    sys.exit(1)

print("Loading face detector...")
try:
    detector = FaceDetector(resource_path("resources/xml/frontal_face.xml"))
    print("✓ Face detector loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load face detector: {e}")
    del webcam
    sys.exit(1)

# Check if folder already exists
if os.path.exists(folder):
    print(f"⚠ Warning: Student '{student_name}' already exists.")
    print(f"   Adding more images to existing folder...")
else:
    print(f"✓ Creating new folder for {student_name}")

# Create folder for this student
os.makedirs(folder, exist_ok=True)

# Setup window - make sure it's visible
window_name = "Save Image - Press Q to quit"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

# Collection parameters
NUM_IMAGES = 10        # Number of images to collect
counter = 1            # Image counter
timer = 0              # Timer for spacing image captures
CAPTURE_INTERVAL = 700 # Milliseconds between captures

print(f"\n{'='*60}")
print(f"READY TO COLLECT {NUM_IMAGES} IMAGES")
print(f"{'='*60}")
print(f"\nInstructions:")
print(f"  • Position face in front of the camera")
print(f"  • Keep face clearly visible")
print(f"  • Images will be captured automatically")
print(f"  • Press 'Q' or close window to stop early")
print(f"\nStarting collection...\n")

# Test camera by showing a frame immediately to open window
print("Opening camera window...")
test_frame = webcam.get_frame()
if test_frame is None:
    print("ERROR: Could not capture frame from camera")
    del webcam
    sys.exit(1)

# Show initial frame to open window
cv2.imshow(window_name, test_frame)
cv2.waitKey(100)  # Brief wait to ensure window opens
print("✓ Camera window opened successfully")

# Main collection loop
while counter <= NUM_IMAGES:
    # Get frame from camera
    frame = webcam.get_frame()
    
    if frame is None:
        print("ERROR: Failed to capture frame")
        break
    
    # Detect faces in frame
    faces_coord = detector.detect(frame)
    
    # Capture image when face is detected and timer condition is met
    if len(faces_coord) > 0 and timer % CAPTURE_INTERVAL == 50:
        # Normalize and process the face
        faces = normalize_faces(frame, faces_coord)
        
        # Save the first detected face
        image_path = f"{folder}/{counter}.jpg"
        cv2.imwrite(image_path, faces[0])
        
        print(f"✓ Image {counter}/{NUM_IMAGES} saved! ({image_path})")
        counter += 1
    
    # Draw rectangle around detected face
    draw_rectangle(frame, faces_coord)
    
    # Add progress text on frame
    progress_text = f"Progress: {counter-1}/{NUM_IMAGES} images captured"
    if len(faces_coord) > 0:
        status_text = "Face detected - Ready to capture"
        color = (0, 255, 0)  # Green
    else:
        status_text = "No face detected - Position face in frame"
        color = (0, 0, 255)  # Red
    
    cv2.putText(frame, progress_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display live feed - this keeps the window open and visible
    cv2.imshow(window_name, frame)
    
    # Wait and update timer - this is essential for window to stay open
    key = cv2.waitKey(50) & 0xFF
    timer += 50
    
    # Break if user presses 'q' or closes window
    if key == ord('q') or key == ord('Q'):
        print("\n⚠ Collection stopped by user")
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("\n⚠ Collection stopped (window closed)")
        break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    if counter > 1:
        print(f"✓ SUCCESS: Collected {counter-1} images for {student_name}")
        print(f"  Images saved in: {folder}/")
        print(f"{'='*60}\n")
    else:
        print(f"⚠ WARNING: No images were collected")
        print(f"  Make sure your face is visible in the camera")
        print(f"{'='*60}\n")

# Release camera
del webcam
print("Camera released. Collection complete.")
