# Student Attendance System with Anti-Spoofing

This system combines face recognition with anti-spoofing technology to create a secure and automated student attendance tracking system.

## Features

- **Liveness Detection**: Uses Silent-Face-Anti-Spoofing to prevent photo/video attacks
- **Face Recognition**: Identifies students and marks attendance automatically
- **Attendance Dashboard**: View, filter, and export attendance records
- **Reports**: Generate visual reports of attendance data
- **Settings**: Configure recognition thresholds

## Setup

### Prerequisites

1. Python 3.7+ installed
2. Webcam for live detection
3. Student face images in the proper directory structure

### Installation

1. Install required dependencies:
```
pip install -r Silent-Face-Anti-Spoofing/requirements.txt
pip install face_recognition opencv-python pandas matplotlib
```



### Running the System

1. Start the application dashboard:
```
python main.py
```

2. Click "Launch Live Attendance System" to start face recognition and attendance tracking

## System Components

- **Silent-Face-Anti-Spoofing**: Detects fake faces using deep learning models
- **Face Recognition**: Identifies students from pre-stored face images
- **Attendance Tracking**: Records recognized students with timestamps
- **Dashboard**: UI for viewing and managing attendance data

## Project Structure

```
StudentAttendanceSystem/
│
├── data/
│   ├── student_images/        # Student face images organized by name
│   └── attendance.csv         # Attendance records (created automatically)
│
├── Silent-Face-Anti-Spoofing/ # Anti-spoofing detection models
│
├── src/
│   ├── face_recognition_attendance.py  # Face recognition module
│   ├── secure_attendance_system.py     # Combines anti-spoofing with recognition
│   └── attendance_dashboard.py         # GUI dashboard for attendance management
│
└── main.py                    # Main entry point for the application
```

## Usage Notes

1. For best accuracy, ensure good lighting when capturing attendance
2. Add multiple images of each student for better recognition
3. Adjust recognition thresholds in settings if needed
4. Keep the anti-spoofing models updated

## Troubleshooting

- If face detection is inaccurate, try adjusting the recognition threshold
- For spoofing detection issues, ensure proper lighting conditions
- If the system is slow, consider reducing the number of models loaded