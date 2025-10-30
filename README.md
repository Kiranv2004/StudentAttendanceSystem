# 🎓 Student Attendance System with Face Recognition, Anti-Spoofing & Emotion Detection

A comprehensive web-based student attendance system that combines **Face Recognition**, **Anti-Spoofing Detection**, and **Emotion Analysis** to provide secure, automated attendance tracking with real-time analytics.

---

## 📋 Table of Contents
- [System Overview](#-system-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Web Pages & Features](#-web-pages--features)
- [Database Schema](#-database-schema)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 System Overview

This project is a **Flask-based web application** that automates student attendance using advanced computer vision and deep learning techniques. It captures student faces via webcam, verifies they are real (not photos/videos), recognizes the student, detects their emotion, and records attendance in a MongoDB database.

### What Makes This System Unique?
- **Triple AI Protection**: Face Recognition + Anti-Spoofing + Emotion Detection
- **Real-time Processing**: Live webcam feed with instant recognition
- **Modern Web Interface**: Responsive Bootstrap 5 UI with real-time stats
- **Complete Management**: Registration, attendance capture, analytics, and reporting
- **Production-Ready**: MongoDB database with GridFS for scalability

---

## ✨ Key Features

### 🔐 Security & Authentication
- **Silent-Face-Anti-Spoofing**: Detects fake faces (photos, videos, masks) using deep learning models
- **Multi-Model Verification**: Uses multiple anti-spoofing models for higher accuracy
- **Liveness Detection**: Real-time verification that prevents presentation attacks

### 👤 Face Recognition
- **128-Dimensional Face Encodings**: Using the `face_recognition` library (based on dlib)
- **Configurable Thresholds**: Adjustable tolerance for recognition accuracy
- **Multiple Face Detection**: Can detect and recognize multiple students simultaneously
- **Optimized Recognition**: Uses HOG (Histogram of Oriented Gradients) for fast detection

### 😊 Emotion Detection
- **DeepFace Integration**: Advanced emotion analysis using deep learning
- **7 Emotions Detected**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- **Real-time Analysis**: Emotion detected during attendance capture
- **Confidence Scores**: Each emotion comes with a confidence percentage

### 📊 Analytics & Reporting
- **Real-time Dashboard**: View attendance stats, department-wise breakdown
- **Date-based Filtering**: Search attendance by date range, student, or department
- **Visual Reports**: Charts and graphs for attendance patterns
- **CSV Export**: Download attendance records for external analysis

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Web UI)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Dashboard │  │Registration│ │Attendance│  │Analytics │        │
│  │ (Home)   │  │  (Add     │  │ (Capture)│  │ (Reports)│        │
│  │          │  │ Students) │  │          │  │          │        │
│  └────┬─────┘  └─────┬────┘  └─────┬────┘  └────┬─────┘       │
└───────┼──────────────┼─────────────┼────────────┼──────────────┘
        │              │             │            │
        │              ▼             ▼            │
        │      ┌─────────────────────────────┐   │
        │      │   FLASK BACKEND (Python)    │   │
        │      │                              │   │
        └─────►│  Routes:                     │◄──┘
               │  - / (dashboard)             │
               │  - /registration             │
               │  - /attendance               │
               │  - /process_attendance       │
               │  - /analytics                │
               │  - /get_attendance           │
               └──────────┬───────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│Face Recognition│ Anti-Spoofing│  │Emotion      │
│                │               │  │Detection    │
│face_recognition│Silent-Face-   │  │             │
│library         │Anti-Spoofing  │  │DeepFace     │
│(dlib-based)    │(PyTorch CNN)  │  │(Keras/TF)   │
│                │               │  │             │
│• 128D encodings│• 2 models     │  │• 7 emotions │
│• HOG detection │• 0.3 threshold│  │• FER2013    │
│• 0.6 tolerance │• Real vs Fake │  │• Confidence │
└────────┬───────┴───────┬───────┴───────┬───────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                ┌─────────────────┐
                │  MongoDB Database│
                │                  │
                │  Collections:    │
                │  - students      │
                │  - attendance    │
                │  - sessions      │
                │                  │
                │  GridFS: Images  │
                └──────────────────┘
```

---

## 🔄 How It Works

### Complete Workflow (Step-by-Step)

#### 1️⃣ **Student Registration**
```
User → Registration Page → Capture Photo → Extract Face Encoding
      → Save to MongoDB → Add to Recognition System
```

**Technical Details:**
- User fills form (Name, USN, Department, Email, etc.)
- Webcam captures student photo
- `face_recognition.face_encodings()` generates 128-dimensional vector
- Face encoding + metadata stored in MongoDB `students` collection
- Image stored in GridFS for reference

#### 2️⃣ **Attendance Capture (Real-time)**
```
Camera Feed → Frame Capture → Face Detection → Anti-Spoofing Check
    → (Pass) → Face Recognition → Emotion Detection → Mark Attendance
    → (Fail) → Reject & Alert User
```

**Technical Details:**

**A. Face Detection**
- OpenCV captures webcam frames (JavaScript → Canvas → Base64 → Flask)
- `face_recognition.face_locations()` finds face bounding boxes
- Uses HOG (Histogram of Oriented Gradients) algorithm
- Returns coordinates: (top, right, bottom, left)

**B. Anti-Spoofing Verification**
- Crops face region from frame
- Patches generated (80x80 pixels) for each face
- Silent-Face-Anti-Spoofing models predict:
  - **Label 0**: Fake (photo/video)
  - **Label 1**: Real (live person)
  - **Label 2**: Spoof (advanced attack)
- Uses 2 pre-trained models:
  - `MiniFASNetV2.pth` (2.7MB)
  - `MiniFASNetV1SE.pth` (4.0MB)
- Threshold: 0.3 (faces scoring < 0.3 are rejected)
- **Result**: Pass/Fail with confidence score

**C. Face Recognition** (Only if anti-spoofing passes)
- Extract face encoding from detected face
- Compare with all stored encodings in database
- Calculate face distance using Euclidean distance
- Match threshold: 0.6 (lower = stricter)
- Returns: Student ID, Name, Confidence score

**D. Emotion Detection** (Optional, if DeepFace available)
- Crop face with 30px padding for context
- Resize to minimum 100x100 for better accuracy
- Convert BGR → RGB (DeepFace requirement)
- `DeepFace.analyze()` predicts emotion:
  - Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- Uses FER2013 emotion model
- Returns dominant emotion + confidence percentage

**E. Database Recording**
- Check if student already marked today (prevent duplicates)
- Save to MongoDB `attendance` collection:
  ```json
  {
    "student_id": "1RV21CS001",
    "student_name": "John Doe",
    "date": "2025-10-30",
    "time": "09:15:30",
    "subject": "Data Structures",
    "class": "CSE-A",
    "emotion": "happy",
    "emotion_confidence": 0.87,
    "is_real": true,
    "anti_spoof_score": 0.92
  }
  ```

#### 3️⃣ **Analytics & Reporting**
```
Database Query → Filter by Date/Student/Department
    → Aggregate Statistics → Generate Charts → Export CSV
```

---

## 🛠️ Technology Stack

### Frontend
- **HTML5/CSS3**: Structure and styling
- **Bootstrap 5**: Responsive UI framework
- **JavaScript (ES6+)**: Real-time camera handling
- **Font Awesome**: Icons
- **Chart.js**: Data visualization (analytics page)

### Backend
- **Python 3.7+**: Core programming language (Built on python 3.9.13 version)
- **Flask 2.x**: Web framework
  - Routes for dashboard, registration, attendance, analytics
  - Session management
  - JSON API endpoints
- **Flask-Compress**: Response compression for faster loading

### AI/ML Libraries
1. **Face Recognition** (`face_recognition` v1.3+)
   - Based on dlib's deep learning face recognition
   - 99.38% accuracy on LFW benchmark
   
2. **Silent-Face-Anti-Spoofing** (Custom)
   - PyTorch-based CNNs (MiniFASNet architectures)
   - Trained on OULU-NPU, SiW, CASIA-FASD datasets
   
3. **DeepFace** (v0.0.75+)
   - Multiple backend support (Keras, TensorFlow, PyTorch)
   - FER2013 emotion model (7 emotions)

4. **OpenCV** (`cv2` v4.5+)
   - Image processing
   - Video capture
   - Face preprocessing

5. **NumPy**: Array operations
6. **SciPy**: Distance calculations

### Database
- **MongoDB 4.x+**: NoSQL database
  - Collections: `students`, `attendance`, `sessions`
  - GridFS: Binary image storage
- **PyMongo**: MongoDB Python driver

### Additional Tools
- **Pandas**: Data manipulation for reports
- **Matplotlib**: Backup visualization (if needed)

---

## 📦 Installation & Setup

### Prerequisites
1. **Python 3.7+** (3.8 recommended)
2. **MongoDB 4.x+** (Community Edition)
3. **Webcam** (for live capture)
4. **CMake** (for dlib compilation)
5. **Visual Studio Build Tools** (Windows) / **build-essential** (Linux)

### Step 1: Clone Repository
```powershell
cd "C:\Users\KIRAN V\Pictures\"
git clone https://github.com/Kiranv2004/StudentAttendanceSystem.git
cd StudentAttendanceSystem
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

**Install Silent-Face-Anti-Spoofing requirements:**
```powershell
pip install -r Silent-Face-Anti-Spoofing\requirements.txt
```

**Install main project requirements:**
```powershell
pip install flask flask-compress pymongo face_recognition opencv-python numpy scipy pandas matplotlib deepface
```

**Optional (for GPU acceleration):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install MongoDB

**Windows:**
1. Download from: https://www.mongodb.com/try/download/community
2. Install with default settings
3. Start MongoDB service:
```powershell
net start MongoDB
```

**Verify MongoDB is running:**
```powershell
mongo --eval "db.version()"
```

### Step 5: Verify Model Files

Ensure these files exist:
```
Silent-Face-Anti-Spoofing/
  resources/
    anti_spoof_models/
      ✅ 2.7_80x80_MiniFASNetV2.pth
      ✅ 4_0_0_80x80_MiniFASNetV1SE.pth
    detection_model/
      ✅ deploy.prototxt
      ✅ Widerface-RetinaFace.caffemodel
```

If missing, download from: [Silent-Face-Anti-Spoofing repository](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

---

## 🚀 Running the Application

### Start the Flask Server
```powershell
python fixed_integrated_attendance_system.py
```

**Output should show:**
```
✅ DeepFace available for emotion detection
✅ Silent-Face-Anti-Spoofing system available
Connecting to MongoDB...
MongoDB connection successful
Found 15 students in database
✅ Successfully loaded 15 face encodings
 * Running on http://127.0.0.1:5000
```

### Access the Web Interface
Open browser and go to: **http://127.0.0.1:5000**

---

## 🌐 Web Pages & Features

### 1. **Dashboard** (`/` - Home Page)
**File:** `templates/dashboard.html`

**Features:**
- **Statistics Cards:**
  - Total Students Registered
  - Today's Attendance Count
  - Department-wise Breakdown
- **Quick Action Buttons:**
  - Take Attendance
  - Register New Student
  - View Analytics
- **Recent Activity Feed**
- **Real-time Updates** (via AJAX)

**Backend Route:** `@app.route('/')`
**Data Source:** MongoDB aggregation queries

---

### 2. **Registration Page** (`/registration`)
**File:** `templates/registration.html`

**Features:**
- **Student Registration Form:**
  - Name, USN (University Seat Number)
  - Department, Year, Section
  - Email, Phone
- **Live Webcam Capture:**
  - Real-time preview
  - Capture button
  - Preview before submission
- **Face Encoding Generation:**
  - Automatic face detection
  - 128D encoding extraction
  - Validation before saving

**Backend Route:** `@app.route('/registration', methods=['GET', 'POST'])`

**Process:**
1. User fills form
2. Captures photo via webcam (JavaScript)
3. Image sent as Base64 to Flask
4. Flask decodes → extracts face encoding
5. Saves to MongoDB with metadata
6. Reloads face recognition system

**Database Operation:**
```python
students_collection.insert_one({
    'usn': usn,
    'name': name,
    'department': dept,
    'year': year,
    'section': section,
    'email': email,
    'phone': phone,
    'face_encoding': encoding.tolist(),  # 128D array
    'image_id': image_id,  # GridFS reference
    'registered_at': datetime.now(),
    'active': True
})
```

---

### 3. **Attendance Page** (`/attendance`)
**File:** `templates/attendance.html`

**Features:**
- **Live Camera Feed:**
  - Real-time video stream
  - Face detection overlay (green boxes)
  - Anti-spoofing status indicator
- **Detection Panel:**
  - Number of faces detected
  - Anti-spoofing results (Real/Fake)
  - Recognition confidence
  - Emotion detected
- **Attendance Controls:**
  - Start/Stop capture
  - Subject/Class selection
  - Manual refresh
- **Real-time Feedback:**
  - Success/Error messages
  - Student name display
  - Duplicate detection warning

**Backend Routes:**
- `@app.route('/attendance')` - Serve page
- `@app.route('/process_attendance', methods=['POST'])` - Process captured frame

**Process Flow:**
```javascript
// Frontend (JavaScript)
1. Capture frame from webcam (every 2 seconds)
2. Convert canvas to Base64
3. Send to /process_attendance endpoint

// Backend (Flask)
4. Decode Base64 → NumPy array
5. Detect faces → Anti-spoofing check
6. Recognize faces → Emotion detection
7. Mark attendance in MongoDB
8. Return JSON response

// Frontend
9. Display results
10. Show success/error message
```

**API Response:**
```json
{
  "success": true,
  "faces": [
    {
      "student_id": "1RV21CS042",
      "student_name": "John Doe",
      "confidence": 0.89,
      "is_real": true,
      "anti_spoof_score": 0.94,
      "emotion": "happy",
      "emotion_confidence": 0.87,
      "already_marked": false
    }
  ],
  "message": "Attendance marked successfully"
}
```

---

### 4. **Analytics Page** (`/analytics`)
**File:** `templates/analytics.html`

**Features:**
- **Advanced Filters:**
  - Date range picker
  - Student search
  - Department filter
  - Class/Section filter
- **Attendance Table:**
  - Student Name, USN, Date, Time
  - Subject, Class, Emotion
  - Sortable columns
  - Pagination
- **Export Options:**
  - Export to CSV
  - Print report
  - PDF generation (if jsPDF available)
- **Statistics Summary:**
  - Total records
  - Attendance percentage
  - Department-wise breakdown

**Backend Route:** `@app.route('/get_attendance')`

**Query Example:**
```python
filters = {
    'date': {'$gte': start_date, '$lte': end_date},
    'department': department,
    'student_id': {'$regex': search_term, '$options': 'i'}
}
attendance_records = db.attendance.find(filters).sort('date', -1)
```

---

## 🗄️ Database Schema

### MongoDB Collections

#### 1. **students** Collection
```json
{
  "_id": ObjectId("..."),
  "usn": "1RV21CS042",
  "name": "John Doe",
  "department": "CSE",
  "year": "3",
  "section": "A",
  "email": "john@example.com",
  "phone": "9876543210",
  "face_encoding": [0.123, -0.456, ...],  // 128 floats
  "image_id": ObjectId("..."),  // GridFS reference
  "registered_at": ISODate("2025-10-30T10:30:00Z"),
  "active": true
}
```

#### 2. **attendance** Collection
```json
{
  "_id": ObjectId("..."),
  "student_id": "1RV21CS042",
  "student_name": "John Doe",
  "department": "CSE",
  "date": "2025-10-30",
  "time": "09:15:30",
  "timestamp": ISODate("2025-10-30T09:15:30Z"),
  "subject": "Data Structures",
  "class": "CSE-A",
  "session_id": "DS_2025_10_30_09",
  "emotion": "happy",
  "emotion_confidence": 0.87,
  "is_real": true,
  "anti_spoof_score": 0.92,
  "confidence": 0.89
}
```

#### 3. **GridFS** (fs.files, fs.chunks)
Stores student images as binary data with chunking for large files.

---

## ⚙️ Configuration

### Recognition Thresholds
**File:** `fixed_integrated_attendance_system.py`

```python
class FixedWebFaceRecognition:
    def __init__(self):
        self.recognition_tolerance = 0.6  # Lower = stricter (0.4-0.7 recommended)
        self.min_confidence = 0.2  # Minimum confidence to accept
        self.max_distance = 0.6  # Maximum face distance threshold
```

### Anti-Spoofing Threshold
```python
threshold = 0.3  # Default threshold
# Increase (0.4-0.5) for fewer false rejections
# Decrease (0.2-0.25) for stricter security
```

### MongoDB Connection
```python
client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
db = client.attendance_system
```

### Flask Configuration
```python
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change for production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
```

---

## 🐛 Troubleshooting

### Issue: "MongoDB connection failed"
**Solution:**
```powershell
# Check MongoDB service status
net start MongoDB

# Or restart MongoDB
net stop MongoDB
net start MongoDB
```

### Issue: "No faces detected" or "Low recognition accuracy"
**Solutions:**
- Ensure good lighting (front-facing, no shadows)
- Camera should be at eye level
- Face should be clearly visible (no glasses/mask during registration)
- Add multiple photos during registration (different angles)
- Adjust `recognition_tolerance` to 0.5 or 0.55

### Issue: "Anti-spoofing always fails"
**Solutions:**
- Check lighting conditions (avoid too bright/dark)
- Lower threshold to 0.25 or 0.2
- Ensure `.pth` model files exist in `Silent-Face-Anti-Spoofing/resources/`
- Use a better quality webcam

### Issue: "DeepFace emotion detection error"
**Solution:**
```powershell
pip install deepface tf-keras
# If TensorFlow conflicts:
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu
pip install tensorflow==2.13.0
```

### Issue: "face_recognition installation fails"
**Windows Solution:**
```powershell
# Install Visual Studio Build Tools
# Then install dlib first:
pip install cmake
pip install dlib
pip install face_recognition
```

### Issue: "Duplicate attendance marked"
**Solution:** System already handles this - checks if student marked today before saving.

### Issue: "High memory usage"
**Solution:**
- Reduce frame capture frequency (increase interval to 3-5 seconds)
- Resize frames to max 800x600 before processing
- Limit number of anti-spoofing models loaded

---

## 📊 Performance Optimization

1. **Frame Processing:**
   - Process every 2-3 seconds (not every frame)
   - Resize large images to 1024px max dimension

2. **Database:**
   - Index frequently queried fields:
     ```javascript
     db.students.createIndex({ "usn": 1 })
     db.attendance.createIndex({ "date": 1, "student_id": 1 })
     ```

3. **Face Recognition:**
   - Use HOG model (faster) instead of CNN
   - Set `number_of_times_to_upsample=1` (default is 1)

4. **Anti-Spoofing:**
   - Can disable if not needed: set `ANTISPOOFING_AVAILABLE = False`

---

## 🔒 Security Best Practices

1. **Change Flask Secret Key:**
   ```python
   app.config['SECRET_KEY'] = 'generate-strong-random-key'
   ```

2. **Restrict File Uploads:**
   - Only allow images (JPG, PNG)
   - Validate file size and dimensions

3. **MongoDB Security:**
   - Enable authentication in production
   - Use username/password in connection string

4. **HTTPS:**
   - Use SSL certificates for production deployment
   - Consider reverse proxy (Nginx + Gunicorn)

---

## 📝 Future Enhancements

- [ ] Add admin authentication/login system
- [ ] SMS/Email notifications for attendance
- [ ] Mobile app (React Native/Flutter)
- [ ] Attendance reports generation (PDF)
- [ ] Integration with Learning Management Systems (LMS)
- [ ] Multi-camera support for large classrooms
- [ ] Attendance API for third-party integrations
- [ ] Dockerized deployment

---

## 👨‍💻 Developer Information

**Repository:** https://github.com/Kiranv2004/StudentAttendanceSystem  
**Developer:** Kiran V  
**Last Updated:** October 30, 2025

---

## 📄 License

This project is for educational purposes. Please respect the licenses of all included libraries:
- Silent-Face-Anti-Spoofing: [Original License](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- face_recognition: MIT License
- DeepFace: MIT License

---

## 🙏 Acknowledgments

- **Silent-Face-Anti-Spoofing** by Minivision AI
- **face_recognition** by Adam Geitgey
- **DeepFace** by Sefik Ilkin Serengil
- Bootstrap & Font Awesome for UI components

---

**For support or questions, please open an issue on GitHub.**