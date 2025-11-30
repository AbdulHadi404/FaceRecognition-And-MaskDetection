# How the Face Recognition & Mask Detection System Works

## 📖 Overview

This document explains how the Face Recognition & Mask Detection system works through a real-world scenario. Follow along as we track a student named "John Doe" through the entire recognition process.

---

## 🎬 Scenario: Tracking John Doe's Recognition

Let's imagine you're an administrator at an educational institute, and you need to track recognition for a student named **John Doe** using face recognition and mask detection.

---

## Step 1: Registering a New Student (collect_images.py)

### Scenario

John Doe is a new student who needs to be registered in the system. You need to capture his face images so the system can recognize him later.

### What Happens

1. **You launch the application** from `main_window.py` and click **"01 Enter New Student"**

2. **The system prompts you** to enter the student's name:

   ```
   Enter student name: john doe
   ```

3. **The system creates a folder** `members/john doe/` to store John's face images

4. **Your webcam activates** and shows a live video feed

5. **John positions himself** in front of the camera

6. **The system automatically captures 10 images**:

   - Detects John's face in real-time using Haar Cascade classifier
   - Waits 700 milliseconds between each capture (to get varied angles/expressions)
   - For each image:
     - Crops the face region
     - Converts to grayscale
     - Applies histogram equalization (improves contrast)
     - Resizes to 100x100 pixels (standard size)
     - Saves as `1.jpg`, `2.jpg`, ... `10.jpg` in `members/john doe/`

7. **You see a preview** of each saved image

8. **Result**: John's face images are now stored and ready for training

### Behind the Scenes

- **Face Detection**: Uses `resources/xml/frontal_face.xml` (Haar Cascade) to detect faces
- **Image Normalization**: All images are processed to ensure consistency
- **Storage**: Images saved in `members/john doe/` directory

---

## Step 2: Training the Recognition Models (train_models.py)

### Scenario

Now that John's images are collected, you need to train the system to recognize him. You also have other students (e.g., "Jane Smith", "Bob Wilson") already in the system.

### What Happens

1. **You click "02 Train Model"** from the main menu

2. **The system scans** the `members/` directory and finds:

   - `members/john doe/` (10 images)
   - `members/jane smith/` (10 images)
   - `members/bob wilson/` (10 images)

3. **The system loads all images**:

   - Reads all images from each person's folder
   - Assigns numeric labels: John=0, Jane=1, Bob=2
   - Creates a mapping: `{0: "john doe", 1: "jane smith", 2: "bob wilson"}`

4. **The system trains recognition algorithms**:

   - **EigenFace**: Uses Principal Component Analysis (PCA) - works with 1+ people
   - **FisherFace**: Uses Linear Discriminant Analysis (LDA) - requires 2+ people (skipped if only 1 person)
   - **LBPH**: Uses Local Binary Patterns Histograms (most robust) - works with 1+ people
   
   **Note**: If only one person is in the system, FisherFace training is automatically skipped since it requires at least 2 classes (people) to function. EigenFace and LBPH will train successfully with just one person.

5. **Training completes** and you see:

   ```
   ✓ All models trained successfully!
   ```

6. **Optional Test**: The system captures a test image from your webcam and shows predictions from all three models

### Behind the Scenes

- **Dataset Loading**: `collect_dataset()` function loads all images and creates labels
- **Model Training**: Each algorithm learns patterns from the face images
- **Label Mapping**: System remembers which number corresponds to which person
- **Note**: Models are trained in memory (not saved to disk) and used immediately

---

## Step 3: Real-Time Recognition Tracking (face_recognition.py)

### Scenario

It's Monday morning, 9:00 AM. John arrives at the institute and walks through the entry gate. Later, at 5:00 PM, he leaves through the exit gate.

### Entry Process (9:00 AM)

1. **You click "03 Face Recognition & Mask Detection"** from the main menu

2. **Two camera windows open**:

   - **Entry Cam** (Camera 0): Monitors the entrance
   - **Exit Cam** (Camera 1): Monitors the exit

3. **John approaches the entry gate**:

   - The Entry Cam detects his face
   - The system also checks if he's wearing a mask

4. **Face Recognition Process**:

   - Detects John's face in the video frame
   - Crops and normalizes the face (same process as Step 1)
   - Runs the face through the trained LBPH model
   - Gets a prediction: "john doe" with confidence score

5. **Mask Detection**:

   - Uses `xml/mask_cascade.xml` to detect if John is wearing a mask
   - Draws a green box around the mask if detected
   - Sets `mask_detected = True` or `False`

6. **Recognition Result**:

   - If confidence is below threshold (76): **Recognized as "John Doe"**
   - If confidence is above threshold: **Shows "Unknown"**

7. **Recognition Record Created**:

   ```
   Name: john doe
   Date: 2024-01-15
   Time: 09:00:00
   Mask: True
   ```

   - Saved to: `records_in/Record_john doe-2024-01-15_09-00-00.csv`

8. **Visual Feedback**:
   - Green rectangle around John's face
   - Text overlay: "John Doe" (in green)
   - Mask indicator: "Using Mask" (if detected)

### Exit Process (5:00 PM)

1. **John approaches the exit gate**:

   - The Exit Cam detects his face
   - Same recognition process runs

2. **Exit Recognition Record Created**:

   ```
   Name: john doe
   Date: 2024-01-15
   Time: 17:00:00
   ```

   - Saved to: `records_out/Record_john doe-2024-01-15_17-00-00.csv`
   - Note: Exit records don't include mask status

3. **Visual Feedback**: Same as entry (green box, name label)

4. **You press ESC** to stop the system when done

### Behind the Scenes

- **Dual Camera Processing**: System processes both cameras simultaneously
- **Real-Time Detection**: Continuously analyzes video frames (30 FPS)
- **LBPH Algorithm**: Used for recognition (most robust to lighting/angle changes)
- **Threshold Logic**: Lower confidence = better match (LBPH specific)
- **CSV Storage**: Each recognition event creates a separate CSV file

---

## Step 4: Consolidating Recognition Records (consolidate_records.py)

### Scenario

At the end of the day, you want to see a complete report of who came in, when they left, and how long they stayed.

### What Happens

1. **You click "04 Save Recognition File"** from the main menu

2. **The system scans** the recognition directories:

   - Reads all CSV files from `records_in/` (entry records)
   - Reads all CSV files from `records_out/` (exit records)

3. **The system finds John's records**:

   - Entry: `Record_john doe-2024-01-15_09-00-00.csv`
   - Exit: `Record_john doe-2024-01-15_17-00-00.csv`

4. **The system merges the records**:

   - Matches entry and exit records by name
   - Combines them into a single row

5. **The system calculates engagement time**:

   - Entry time: 09:00:00
   - Exit time: 17:00:00
   - Difference: 8 hours = 480 minutes

6. **Final consolidated record**:

   ```
   Name: john doe
   DateIn: 2024-01-15
   TimeIn: 09:00:00
   Mask: True
   DateOut: 2024-01-15
   TimeOut: 17:00:00
   Engage-Min: 480
   Engage-Hrs: 8.00
   ```

7. **Report saved** to: `recognition_results/Recognition_Result_2024-01-15.csv`

### Behind the Scenes

- **File Reading**: Uses `glob` to find all CSV files
- **Data Merging**: Uses pandas `merge()` function (left join on 'Name')
- **Time Calculation**: Converts time strings to datetime, calculates difference
- **Consolidation**: All students' records are combined into one report

---

## 🔄 Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. COLLECT IMAGES                                           │
│    └─> Capture 10 face images → members/john doe/          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAIN MODELS                                             │
│    └─> Load all images → Train 3 algorithms → Ready!      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. DETECT & RECOGNIZE (Real-time)                           │
│    Entry: Detect face → Recognize → Save entry record        │
│    Exit:  Detect face → Recognize → Save exit record          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CONSOLIDATE                                               │
│    └─> Merge entry + exit → Calculate time → Final report  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Concepts Explained

### Face Detection vs. Face Recognition

- **Face Detection**: "There's a face in this image" (uses Haar Cascade)
- **Face Recognition**: "This face belongs to John Doe" (uses trained models)

### Why 10 Images?

- Captures different angles, expressions, and lighting conditions
- Improves recognition accuracy
- More training data = better model performance

### Recognition Algorithms

- **EigenFace**: Fast, good for controlled environments (works with 1+ people)
- **FisherFace**: Better at handling variations (requires 2+ people)
- **LBPH**: Most robust, handles lighting/angle changes best (used in production, works with 1+ people)

**Note**: FisherFace requires at least 2 people in the dataset. If only one person is registered, the system will automatically skip FisherFace training and only train EigenFace and LBPH.

### Confidence Thresholds

- **LBPH**: Lower values = better match (threshold: 75-76)
- If confidence > threshold → "Unknown"
- If confidence ≤ threshold → Recognized person

### Mask Detection

- Uses a separate Haar Cascade classifier (`resources/xml/mask_cascade.xml`)
- Detects if a person is wearing a face mask
- Only tracked at entry (not exit)

### Camera System

- **Camera 0**: Default webcam for recognition tracking
- Tracks when people arrive + mask status
- Real-time face recognition and logging

---

## 📊 Data Flow

```
Video Frame
    ↓
Face Detection (Haar Cascade)
    ↓
Face Crop & Normalize
    ↓
Face Recognition (LBPH Model)
    ↓
Confidence Check
    ↓
[If Recognized] → Save to CSV
    ↓
Consolidate All Records
    ↓
Final Report
```

---

## 🛠️ Technical Details

### Image Processing Pipeline

1. **Capture**: Get frame from webcam
2. **Detect**: Find face coordinates using Haar Cascade
3. **Crop**: Extract face region (remove 30% from edges)
4. **Normalize**: Convert to grayscale + histogram equalization
5. **Resize**: Scale to 100x100 pixels
6. **Recognize**: Compare with trained models
7. **Result**: Get person name + confidence score

### File Structure

```
FaceRecognition-And-MaskDetection/
├── src/                        # Source code modules
│   ├── collect_images.py
│   ├── train_models.py
│   ├── face_recognition.py
│   └── consolidate_records.py
├── resources/                  # Static resources
│   ├── xml/
│   │   ├── frontal_face.xml
│   │   └── mask_cascade.xml
│   └── images/
├── members/                    # Runtime data (created automatically)
│   ├── john doe/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ... (10 images)
│   └── jane smith/
│       └── ...
├── records_in/
│   └── Record_john doe-2024-01-15_09-00-00.csv
├── records_out/
│   └── Record_john doe-2024-01-15_17-00-00.csv
├── recognition_results/
│   └── Recognition_Result_2024-01-15.csv
└── resources/
    └── xml/
        ├── frontal_face.xml (face detection)
        └── mask_cascade.xml (mask detection)
```

---

## 💡 Tips for Best Results

1. **Good Lighting**: Ensure faces are well-lit for better recognition
2. **Clear View**: Keep face clearly visible to camera
3. **Consistent Environment**: Similar lighting conditions help
4. **Multiple Angles**: The 10 training images should include varied angles
5. **Regular Updates**: Retrain models when adding new students

---

## ❓ Common Questions

**Q: What if someone isn't recognized?**

- A: They'll be marked as "Unknown". Make sure they're registered first.

**Q: Can the system handle multiple people at once?**

- A: Yes! The system can detect and recognize multiple faces simultaneously.

**Q: What happens if someone enters but doesn't exit?**

- A: The consolidation will show their entry time but no exit time (engagement time = 0).

**Q: How accurate is the recognition?**

- A: Depends on lighting, angle, and training quality. LBPH is typically 85-95% accurate.

**Q: Can I use just one camera?**

- A: Yes, but you'll need to modify the code to use only one camera index.

---

## 🎓 Learning Resources

- **Haar Cascade**: Machine learning-based object detection
- **LBPH**: Local Binary Patterns Histograms for face recognition
- **OpenCV**: Computer vision library used throughout
- **Pandas**: Data manipulation for recognition records

---

_This system provides an automated, contactless face recognition and mask detection solution perfect for educational institutes, offices, or any facility requiring identity verification and mask monitoring._
