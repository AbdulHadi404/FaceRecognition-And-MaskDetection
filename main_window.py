"""
Main GUI Application - Face Recognition & Mask Detection System
===============================================================

This is the main entry point for the Face Recognition and Mask Detection application.
It provides a graphical user interface with buttons to access all system features.

Features:
- Enter New Person: Collect face images for training
- Train Model: Train face recognition models
- Face Recognition & Mask Detection: Real-time face recognition and mask detection
"""

import sys
import os
import subprocess
import threading
import time
from tkinter import *
from tkinter import font
from tkinter.font import BOLD
import tkinter.messagebox as messagebox
from tkinter import simpledialog
from PIL import Image, ImageTk

# Setup logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from logger_setup import setup_logger
logger = setup_logger('main_window')

# Import camera configuration
try:
    from config.camera_config import get_camera_source, save_camera_config, load_camera_config
    CAMERA_CONFIG_AVAILABLE = True
except ImportError:
    CAMERA_CONFIG_AVAILABLE = False
    logger.warning("Camera configuration module not available")
    def get_camera_source():
        return 0
    def save_camera_config(source):
        pass

logger.info("="*60)
logger.info("Main Window Application Started")
logger.info("="*60)
logger.debug(f"Python executable: {sys.executable}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Running as executable: {getattr(sys, 'frozen', False)}")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Window settings
WINDOW_TITLE = 'Face Recognition & Mask Detection'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Color scheme (modern dark theme)
BG_COLOR = "#2C3E50"          # Dark blue-gray background
ACCENT_COLOR = "#3498DB"      # Blue accent
BUTTON_COLOR = "#3498DB"      # Button color
BUTTON_HOVER = "#2980B9"      # Button hover color
TEXT_COLOR = "#ECF0F1"        # Light text color
FRAME_COLOR = "#34495E"       # Frame background color
SUCCESS_COLOR = "#27AE60"     # Green for success messages
WARNING_COLOR = "#F39C12"     # Orange for warnings


# ============================================================================
# MAIN WINDOW SETUP
# ============================================================================

root = Tk()
root.title(WINDOW_TITLE)
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.resizable(False, False)
root.config(bg=BG_COLOR)

# Try to load background image (optional)
try:
    img = ImageTk.PhotoImage(file="resources/images/img.jpg")
    lab = Label(root, image=img, bg=BG_COLOR)
    lab.place(x=0, y=0, relwidth=1, relheight=1)
except:
    pass  # Continue without background image if it doesn't exist


# ============================================================================
# STATUS MESSAGE DISPLAY
# ============================================================================

# Status label at the bottom
status_frame = Frame(root, bg=BG_COLOR)
status_frame.pack(side=BOTTOM, fill=X, padx=10, pady=5)

status_label = Label(
    status_frame,
    text="Ready",
    font=('Helvetica', 10),
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    anchor=W
)
status_label.pack(side=LEFT, fill=X, expand=True)


def update_status(message, color=TEXT_COLOR):
    """Update the status message at the bottom of the window."""
    status_label.config(text=message, fg=color)
    root.update_idletasks()

def reset_status_after_delay(delay=3):
    """Reset status to 'Ready' after a delay."""
    root.after(delay * 1000, lambda: update_status("Ready", TEXT_COLOR))

def monitor_process_completion(process, success_message, error_message=None):
    """Monitor a process and update status when it completes."""
    def monitor():
        process.wait()  # Wait for process to complete
        # Small delay to let GUI dialogs in subprocess complete
        time.sleep(1)
        if process.returncode == 0:
            update_status(success_message, SUCCESS_COLOR)
            # Reset to ready after showing success for a few seconds
            root.after(4000, lambda: update_status("Ready", TEXT_COLOR))
        else:
            if error_message:
                update_status(error_message, WARNING_COLOR)
            else:
                update_status("Process completed with errors", WARNING_COLOR)
            # Reset after showing error
            root.after(4000, lambda: update_status("Ready", TEXT_COLOR))
    threading.Thread(target=monitor, daemon=True).start()


# ============================================================================
# BUTTON COMMAND FUNCTIONS
# ============================================================================

def collect():
    """Launch the image collection module to capture face images."""
    # Get person name from GUI dialog
    person_name = simpledialog.askstring(
        "Enter Person Name",
        "Please enter the person's name:",
        parent=root
    )
    
    if person_name is None or person_name.strip() == "":
        messagebox.showwarning("No Name Entered", "Please enter a valid name.")
        update_status("Operation cancelled", WARNING_COLOR)
        return
    
    person_name = person_name.strip()
    
    # Check if person already exists
    folder = f"members/{person_name.lower()}"
    if os.path.exists(folder):
        response = messagebox.askyesno(
            "Person Already Exists",
            f"'{person_name}' already exists.\n\nDo you want to add more images?",
            parent=root
        )
        if not response:
            update_status("Operation cancelled", WARNING_COLOR)
            return
    
    update_status(f"Starting image collection for {person_name}...", ACCENT_COLOR)
    
    # Launch collection script with person name as argument
    try:
        logger.info(f"Launching image collection for: {person_name}")
        script_path = os.path.join('src', 'collect_images.py')
        logger.debug(f"Script path: {script_path}")
        logger.debug(f"Script exists: {os.path.exists(script_path)}")
        
        process = subprocess.Popen(['python', script_path, person_name])
        logger.info(f"Image collection subprocess started with PID: {process.pid}")
        update_status(f"Collecting images for {person_name}... Camera window opening...", ACCENT_COLOR)
        # Monitor process completion
        monitor_process_completion(
            process, 
            f"✓ Image collection completed for {person_name}",
            f"Image collection failed for {person_name}"
        )
    except Exception as e:
        error_msg = f"Failed to start image collection: {e}"
        logger.exception(error_msg)
        messagebox.showerror("Error", f"Failed to start image collection:\n{str(e)}", parent=root)
        update_status("Error starting image collection", WARNING_COLOR)
        reset_status_after_delay(3)


def train():
    """Launch the model training module to train face recognition models."""
    # Check if members directory exists and has data
    if not os.path.exists("members"):
        messagebox.showwarning(
            "No Data Found",
            "No face images found!\n\n"
            "Please collect images first using 'Enter New Person'.",
            parent=root
        )
        update_status("No data to train", WARNING_COLOR)
        return
    
    members = [d for d in os.listdir("members") if os.path.isdir(os.path.join("members", d))]
    if len(members) == 0:
        messagebox.showwarning(
            "No Data Found",
            "No face images found!\n\n"
            "Please collect images first using 'Enter New Person'.",
            parent=root
        )
        update_status("No data to train", WARNING_COLOR)
        return
    
    update_status(f"Training models for {len(members)} person(s)... Please wait.", ACCENT_COLOR)
    
    try:
        logger.info("Launching model training")
        script_path = os.path.join('src', 'train_models.py')
        logger.debug(f"Script path: {script_path}")
        
        process = subprocess.Popen(['python', script_path])
        logger.info(f"Training subprocess started with PID: {process.pid}")
        update_status("Training in progress...", ACCENT_COLOR)
        # Monitor process completion
        monitor_process_completion(
            process,
            "✓ Model training completed successfully",
            "Model training failed"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start training:\n{str(e)}", parent=root)
        update_status("Error starting training", WARNING_COLOR)
        reset_status_after_delay(3)


def recognize():
    """Launch the real-time face recognition and mask detection module."""
    # Check if models are trained (check if members directory exists)
    if not os.path.exists("members"):
        messagebox.showwarning(
            "No Training Data",
            "No face data found!\n\n"
            "Please:\n"
            "1. Collect images (Enter New Person)\n"
            "2. Train models (Train Model)\n\n"
            "Then try again.",
            parent=root
        )
        update_status("No training data available", WARNING_COLOR)
        return
    
    update_status("Starting face recognition system...", ACCENT_COLOR)
    
    try:
        logger.info("Launching face recognition")
        script_path = os.path.join('src', 'face_recognition.py')
        logger.debug(f"Script path: {script_path}")
        logger.debug(f"Script exists: {os.path.exists(script_path)}")
        logger.debug(f"Members directory exists: {os.path.exists('members')}")
        
        process = subprocess.Popen(['python', script_path])
        logger.info(f"Face recognition subprocess started with PID: {process.pid}")
        update_status("Face recognition active - Camera window opening...", SUCCESS_COLOR)
        monitor_process_completion(
            process,
            "Face recognition stopped",
            "Face recognition error"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start recognition:\n{str(e)}", parent=root)
        update_status("Error starting recognition", WARNING_COLOR)


# ============================================================================
# CAMERA CONFIGURATION FUNCTIONS
# ============================================================================

def configure_ip_camera():
    """Open dialog to configure IP camera URL."""
    if not CAMERA_CONFIG_AVAILABLE:
        messagebox.showerror("Error", "Camera configuration module not available", parent=root)
        return
    
    # Get current IP camera URL if set
    current_source = get_camera_source()
    current_url = current_source if isinstance(current_source, str) else "http://192.168.1.8:8080/video"
    
    # Ask for IP camera URL
    url = simpledialog.askstring(
        "Configure IP Camera",
        "Enter IP Camera URL:\n\nExample: http://192.168.1.8:8080/video\nOr: rtsp://192.168.1.8:8086/stream",
        initialvalue=current_url,
        parent=root
    )
    
    if url:
        url = url.strip()
        if url.startswith("http://") or url.startswith("https://") or url.startswith("rtsp://"):
            save_camera_config(url)
            update_status(f"IP Camera configured: {url}", SUCCESS_COLOR)
            # Update radio button state to IP camera mode
            if hasattr(root, 'camera_mode_var'):
                root.camera_mode_var.set("ip")
            update_camera_status()
            root.after(2000, lambda: update_status("Ready", TEXT_COLOR))
        else:
            messagebox.showerror("Invalid URL", "URL must start with http://, https://, or rtsp://", parent=root)

def switch_camera_mode():
    """Switch between webcam and IP camera based on radio button selection."""
    if not CAMERA_CONFIG_AVAILABLE:
        return
    
    if not hasattr(root, 'camera_mode_var'):
        return
    
    mode = root.camera_mode_var.get()
    
    if mode == "webcam":
        # Switch to local webcam (index 0)
        save_camera_config(0)
        update_status("Switched to local webcam", SUCCESS_COLOR)
        update_camera_status()
        root.after(2000, lambda: update_status("Ready", TEXT_COLOR))
    elif mode == "ip":
        # Switch to IP camera - use current config or prompt
        current_source = get_camera_source()
        if isinstance(current_source, str):
            # Already configured
            update_status(f"Using IP Camera: {current_source}", SUCCESS_COLOR)
            update_camera_status()
            root.after(2000, lambda: update_status("Ready", TEXT_COLOR))
        else:
            # Not configured, prompt user
            configure_ip_camera()
            update_camera_status()


# ============================================================================
# UI COMPONENTS
# ============================================================================

# Main container frame
main_frame = Frame(root, bg=BG_COLOR)
main_frame.pack(expand=True, fill=BOTH, padx=40, pady=40)

# Title section
title_frame = Frame(main_frame, bg=BG_COLOR)
title_frame.pack(pady=(0, 30))

title_label = Label(
    title_frame, 
    text="Face Recognition & Mask Detection", 
    font=('Helvetica', 24, 'bold'),
    bg=BG_COLOR,
    fg=TEXT_COLOR
)
title_label.pack()

subtitle_label = Label(
    title_frame,
    text="Real-time Face Recognition & Mask Detection",
    font=('Helvetica', 12),
    bg=BG_COLOR,
    fg="#BDC3C7"
)
subtitle_label.pack(pady=(5, 0))

# Camera settings frame
camera_frame = Frame(main_frame, bg=BG_COLOR)
camera_frame.pack(pady=(0, 20))

# Camera mode label
camera_label = Label(
    camera_frame,
    text="Camera Source:",
    font=('Helvetica', 10, 'bold'),
    bg=BG_COLOR,
    fg=TEXT_COLOR
)
camera_label.pack(side=LEFT, padx=(0, 10))

# Camera mode variable
root.camera_mode_var = StringVar(value="webcam")

# Determine initial camera mode
if CAMERA_CONFIG_AVAILABLE:
    try:
        current_source = get_camera_source()
        if isinstance(current_source, str):
            root.camera_mode_var.set("ip")
    except:
        pass

# Webcam radio button
webcam_radio = Radiobutton(
    camera_frame,
    text="Local Webcam",
    variable=root.camera_mode_var,
    value="webcam",
    font=('Helvetica', 10),
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    selectcolor=FRAME_COLOR,
    activebackground=BG_COLOR,
    activeforeground=TEXT_COLOR,
    command=switch_camera_mode
)
webcam_radio.pack(side=LEFT, padx=5)

# IP Camera radio button
ip_camera_radio = Radiobutton(
    camera_frame,
    text="IP Camera",
    variable=root.camera_mode_var,
    value="ip",
    font=('Helvetica', 10),
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    selectcolor=FRAME_COLOR,
    activebackground=BG_COLOR,
    activeforeground=TEXT_COLOR,
    command=switch_camera_mode
)
ip_camera_radio.pack(side=LEFT, padx=5)

# Configure IP Camera button
config_ip_btn = Button(
    camera_frame,
    text="Configure IP",
    command=configure_ip_camera,
    font=('Helvetica', 9),
    bg=FRAME_COLOR,
    fg=TEXT_COLOR,
    activebackground="#2C3E50",
    activeforeground=TEXT_COLOR,
    relief=FLAT,
    padx=10,
    pady=2,
    cursor='hand2'
)
config_ip_btn.pack(side=LEFT, padx=(15, 0))

# Current camera status label
camera_status_label = Label(
    camera_frame,
    text="",
    font=('Helvetica', 9),
    bg=BG_COLOR,
    fg="#BDC3C7"
)
camera_status_label.pack(side=LEFT, padx=(10, 0))

# Update camera status display
def update_camera_status():
    """Update the camera status label."""
    if CAMERA_CONFIG_AVAILABLE:
        try:
            current_source = get_camera_source()
            if isinstance(current_source, str):
                # Show shortened URL
                display_url = current_source if len(current_source) < 40 else current_source[:37] + "..."
                camera_status_label.config(text=f"[{display_url}]")
            else:
                camera_status_label.config(text=f"[Webcam {current_source}]")
        except:
            camera_status_label.config(text="")
    else:
        camera_status_label.config(text="[Not Available]")

# Initial status update
update_camera_status()

# Menu frame with styling
menu_frame = Frame(main_frame, bg=FRAME_COLOR, relief=RAISED, bd=2)
menu_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)

# Menu title
menu_title = Label(
    menu_frame,
    text="Main Menu",
    font=('Helvetica', 16, 'bold'),
    bg=FRAME_COLOR,
    fg=TEXT_COLOR
)
menu_title.pack(pady=(20, 30))

# Button style configuration (applied to all buttons)
button_style = {
    'font': ('Helvetica', 11, 'bold'),
    'bg': BUTTON_COLOR,
    'fg': TEXT_COLOR,
    'activebackground': BUTTON_HOVER,
    'activeforeground': TEXT_COLOR,
    'relief': RAISED,
    'bd': 2,
    'cursor': 'hand2',
    'width': 35,
    'height': 2
}

# Create menu buttons
button_collect = Button(
    menu_frame,
    text="01  Enter New Person",
    command=collect,
    **button_style
)
button_collect.pack(pady=8, padx=30)

button_train = Button(
    menu_frame,
    text="02  Train Model",
    command=train,
    **button_style
)
button_train.pack(pady=8, padx=30)

button_recognize = Button(
    menu_frame,
    text="03  Face Recognition & Mask Detection",
    command=recognize,
    **button_style
)
button_recognize.pack(pady=8, padx=30)


# ============================================================================
# BUTTON HOVER EFFECTS
# ============================================================================

def on_enter(button):
    """Change button color when mouse enters."""
    button.config(bg=BUTTON_HOVER)

def on_leave(button):
    """Restore button color when mouse leaves."""
    button.config(bg=BUTTON_COLOR)

# Apply hover effects to all buttons
for btn in [button_collect, button_train, button_recognize]:
    btn.bind("<Enter>", lambda e, b=btn: on_enter(b))
    btn.bind("<Leave>", lambda e, b=btn: on_leave(b))


# ============================================================================
# START APPLICATION
# ============================================================================

if __name__ == "__main__":
    root.mainloop()
