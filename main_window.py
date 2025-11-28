"""
Main GUI Application - Face Recognition & Mask Detection System
===============================================================

This is the main entry point for the Face Recognition and Mask Detection application.
It provides a graphical user interface with buttons to access all system features.

Features:
- Enter New Student: Collect face images for training
- Train Model: Train face recognition models
- Face Attendance & Mask Detection: Real-time attendance tracking
- Save Face Attendance File: Consolidate attendance records
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
    """Launch the image collection module to capture student face images."""
    # Get student name from GUI dialog
    student_name = simpledialog.askstring(
        "Enter Student Name",
        "Please enter the student's name:",
        parent=root
    )
    
    if student_name is None or student_name.strip() == "":
        messagebox.showwarning("No Name Entered", "Please enter a valid student name.")
        update_status("Operation cancelled", WARNING_COLOR)
        return
    
    student_name = student_name.strip()
    
    # Check if student already exists
    folder = f"members/{student_name.lower()}"
    if os.path.exists(folder):
        response = messagebox.askyesno(
            "Student Already Exists",
            f"Student '{student_name}' already exists.\n\nDo you want to add more images?",
            parent=root
        )
        if not response:
            update_status("Operation cancelled", WARNING_COLOR)
            return
    
    update_status(f"Starting image collection for {student_name}...", ACCENT_COLOR)
    
    # Launch collection script with student name as argument
    try:
        process = subprocess.Popen(['python', 'src/collect_images.py', student_name])
        update_status(f"Collecting images for {student_name}... Camera window opening...", ACCENT_COLOR)
        # Monitor process completion
        monitor_process_completion(
            process, 
            f"✓ Image collection completed for {student_name}",
            f"Image collection failed for {student_name}"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start image collection:\n{str(e)}", parent=root)
        update_status("Error starting image collection", WARNING_COLOR)
        reset_status_after_delay(3)


def train():
    """Launch the model training module to train face recognition models."""
    # Check if members directory exists and has data
    if not os.path.exists("members"):
        messagebox.showwarning(
            "No Data Found",
            "No student images found!\n\n"
            "Please collect images first using 'Enter New Student'.",
            parent=root
        )
        update_status("No data to train", WARNING_COLOR)
        return
    
    members = [d for d in os.listdir("members") if os.path.isdir(os.path.join("members", d))]
    if len(members) == 0:
        messagebox.showwarning(
            "No Data Found",
            "No student images found!\n\n"
            "Please collect images first using 'Enter New Student'.",
            parent=root
        )
        update_status("No data to train", WARNING_COLOR)
        return
    
    update_status(f"Training models for {len(members)} student(s)... Please wait.", ACCENT_COLOR)
    
    try:
        process = subprocess.Popen(['python', 'src/train_models.py'])
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
            "No student data found!\n\n"
            "Please:\n"
            "1. Collect images (Enter New Student)\n"
            "2. Train models (Train Model)\n\n"
            "Then try again.",
            parent=root
        )
        update_status("No training data available", WARNING_COLOR)
        return
    
    update_status("Starting face recognition system...", ACCENT_COLOR)
    
    try:
        process = subprocess.Popen(['python', 'src/face_recognition.py'])
        update_status("Face recognition active - Camera window opening...", SUCCESS_COLOR)
        monitor_process_completion(
            process,
            "Face recognition stopped",
            "Face recognition error"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start recognition:\n{str(e)}", parent=root)
        update_status("Error starting recognition", WARNING_COLOR)


def consolidate():
    """Launch the attendance consolidation module to merge and save records."""
    # Check if attendance files exist
    has_attendance = os.path.exists("attendance_in") and len(os.listdir("attendance_in")) > 0
    
    if not has_attendance:
        messagebox.showwarning(
            "No Attendance Data",
            "No attendance records found!\n\n"
            "Please run 'Face Attendance & Mask Detection' first to generate records.",
            parent=root
        )
        update_status("No attendance data to consolidate", WARNING_COLOR)
        return
    
    # Show confirmation
    attendance_count = len(os.listdir("attendance_in"))
    
    response = messagebox.askyesno(
        "Consolidate Attendance",
        f"Ready to consolidate attendance records.\n\n"
        f"Found {attendance_count} attendance record(s)\n\n"
        f"This will process all records and generate a summary report.\n\n"
        f"Continue?",
        parent=root
    )
    
    if not response:
        update_status("Consolidation cancelled", WARNING_COLOR)
        return
    
    update_status("Consolidating attendance records...", ACCENT_COLOR)
    
    try:
        process = subprocess.Popen(['python', 'src/consolidate_attendance.py'])
        update_status("Consolidation in progress...", ACCENT_COLOR)
        # Monitor process completion
        monitor_process_completion(
            process,
            "✓ Attendance consolidation completed",
            "Attendance consolidation failed"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start consolidation:\n{str(e)}", parent=root)
        update_status("Error starting consolidation", WARNING_COLOR)
        reset_status_after_delay(3)


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
    text="Attendance Management System",
    font=('Helvetica', 12),
    bg=BG_COLOR,
    fg="#BDC3C7"
)
subtitle_label.pack(pady=(5, 0))

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
    text="01  Enter New Student",
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
    text="03  Face Attendance & Mask Detection",
    command=recognize,
    **button_style
)
button_recognize.pack(pady=8, padx=30)

button_consolidate = Button(
    menu_frame,
    text="04  Save Face Attendance File",
    command=consolidate,
    **button_style
)
button_consolidate.pack(pady=8, padx=30)


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
for btn in [button_collect, button_train, button_recognize, button_consolidate]:
    btn.bind("<Enter>", lambda e, b=btn: on_enter(b))
    btn.bind("<Leave>", lambda e, b=btn: on_leave(b))


# ============================================================================
# START APPLICATION
# ============================================================================

if __name__ == "__main__":
    root.mainloop()
