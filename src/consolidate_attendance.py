"""
Attendance Consolidation Module - Face Recognition & Mask Detection System
===========================================================================

This module consolidates attendance records into a single report.
It processes all attendance records and generates a summary.

Workflow:
1. Reads all CSV files from attendance_in/ directory
2. Groups records by person and date
3. Generates summary report
4. Saves consolidated report to attendance_results/ directory

Output:
- Consolidated CSV file with columns:
  - Name: Person name
  - Date: Attendance date
  - Time: Attendance time
  - Mask: Mask status
  - Count: Number of attendance records

Requirements:
- attendance_in/ directory with CSV files
"""

import pandas as pd
import glob
import os
import sys
import time

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

from gui_messages import show_info, show_warning, show_error, show_progress


# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory paths
ATTENDANCE_DIR = 'attendance_in'      # Directory containing attendance files
RESULT_DIR = 'attendance_results'      # Directory for consolidated results

# Get current date for filename
current_date = time.strftime('%Y-%m-%d')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_csv_files(directory):
    """
    Load and concatenate all CSV files from a directory.
    
    Args:
        directory: Path to directory containing CSV files
        
    Returns:
        DataFrame containing all concatenated data, or None if no files found
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist.")
        return None
    
    # Find all CSV files in directory
    csv_pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if len(csv_files) == 0:
        print(f"Warning: No CSV files found in '{directory}'")
        return None
    
    # Read and concatenate all CSV files
    dataframes = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if len(dataframes) == 0:
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


# ============================================================================
# MAIN PROCESSING
# ============================================================================

# Create result directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)

# Collect messages for progress window
progress_messages = []
progress_messages.append("=" * 60)
progress_messages.append("ATTENDANCE CONSOLIDATION")
progress_messages.append("=" * 60)

# ============================================================================
# STEP 1: Load Attendance Records
# ============================================================================

progress_messages.append(f"\n[Step 1] Loading attendance records from '{ATTENDANCE_DIR}/'...")
attendance_df = load_csv_files(ATTENDANCE_DIR)

if attendance_df is None or attendance_df.empty:
    show_error(
        "No Attendance Data",
        "No attendance records found!\n\n"
        "Please run 'Face Attendance & Mask Detection' first to generate records."
    )
    sys.exit(1)

progress_messages.append(f"✓ Loaded {len(attendance_df)} attendance records")
progress_messages.append(f"\nAttendance Preview:")
progress_messages.append(str(attendance_df.head()))

# ============================================================================
# STEP 2: Group and Summarize Records
# ============================================================================

progress_messages.append("\n[Step 2] Processing attendance records...")

# Sort by Name, Date, and Time
attendance_df = attendance_df.sort_values(['Name', 'Date', 'Time'])

# Group by Name and Date to get summary
summary = attendance_df.groupby(['Name', 'Date']).agg({
    'Time': ['first', 'last', 'count'],  # First time, last time, count
    'Mask': 'first'  # First mask status
}).reset_index()

# Flatten column names
summary.columns = ['Name', 'Date', 'FirstTime', 'LastTime', 'Count', 'Mask']

# Rename for clarity
result = summary.copy()
result.columns = ['Name', 'Date', 'FirstTime', 'LastTime', 'RecordCount', 'Mask']

progress_messages.append(f"✓ Processed {len(result)} unique person-date combinations")

# ============================================================================
# STEP 3: Display Results and Save
# ============================================================================

progress_messages.append("\n[Step 3] Final consolidated attendance:")
progress_messages.append("=" * 60)
progress_messages.append(result.to_string(index=False))
progress_messages.append("=" * 60)

# Save consolidated result
output_filename = f"{RESULT_DIR}/Attendance_Result_{current_date}.csv"
result.to_csv(output_filename, index=True, header=True)

progress_messages.append(f"\n✓ Consolidated attendance saved to: {output_filename}")
progress_messages.append(f"  Total unique records: {len(result)}")
progress_messages.append(f"  Total attendance entries: {len(attendance_df)}")
progress_messages.append(f"  Date: {current_date}")
progress_messages.append("\n✓ Consolidation complete!")

# Show results in GUI
show_progress("Attendance Consolidation Results", progress_messages)

# Show success message
show_info(
    "Consolidation Complete",
    f"Successfully consolidated attendance records!\n\n"
    f"Total records: {len(result)}\n"
    f"Saved to: {output_filename}"
)
