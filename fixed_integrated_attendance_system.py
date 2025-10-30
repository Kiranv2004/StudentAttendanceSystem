"""
Fixed Integrated Student Attendance System with Web Interface
This fixes all the issues with the web-based attendance system
"""

import os
import cv2
import sys
import time
import numpy as np
import face_recognition
import torch
import torch.nn.functional as F
from datetime import datetime
import json
import base64
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session, flash
from flask_compress import Compress
import pandas as pd
from pymongo import MongoClient
from gridfs import GridFS
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
import threading
import webbrowser

# Optional: DeepFace for emotion analysis
EMOTION_AVAILABLE = False
try:
    from deepface import DeepFace  # type: ignore
    EMOTION_AVAILABLE = True
    print("✅ DeepFace available for emotion detection")
except Exception as _e:
    print(f"⚠️ DeepFace not available: {_e}")

# Simple emotion detection fallback using facial landmarks
def detect_emotion_simple(face_crop):
    """Simple emotion detection based on facial features"""
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristic-based emotion detection
        # This is a basic implementation - in practice, you'd use more sophisticated methods
        
        # Calculate brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Simple rules based on image characteristics
        if brightness > 120 and contrast > 30:
            return "happy", 0.7
        elif brightness < 80 and contrast < 20:
            return "sad", 0.6
        elif contrast > 40:
            return "angry", 0.5
        else:
            return "neutral", 0.4
            
    except Exception as e:
        print(f"Simple emotion detection error: {e}")
        return "neutral", 0.3

# Add Silent-Face-Anti-Spoofing to path
silent_face_dir = os.path.join(os.path.dirname(__file__), 'Silent-Face-Anti-Spoofing')
silent_face_src_dir = os.path.join(silent_face_dir, 'src')
sys.path.insert(0, silent_face_src_dir)

# Try to import Silent-Face-Anti-Spoofing modules
ANTISPOOFING_AVAILABLE = False
try:
    # Add Silent-Face-Anti-Spoofing root directory to Python path
    # This allows the 'src' imports to work properly
    if silent_face_dir not in sys.path:
        sys.path.insert(0, silent_face_dir)
    
    # Import Silent-Face-Anti-Spoofing components
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
    ANTISPOOFING_AVAILABLE = True
    print("✅ Silent-Face-Anti-Spoofing system available")
except ImportError as e:
    print(f"⚠️ Silent-Face-Anti-Spoofing not available: {e}")
    print("Face recognition will work without anti-spoofing")
except Exception as e:
    print(f"⚠️ Silent-Face-Anti-Spoofing initialization error: {e}")
    print("Face recognition will work without anti-spoofing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FIXED FACE RECOGNITION SYSTEM
# =============================================================================

class FixedWebFaceRecognition:
    """Fixed face recognition system for web interface"""
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.known_face_metadata = []
        
        # Recognition parameters - Optimized for better detection
        self.recognition_tolerance = 0.6  # face_recognition library tolerance
        self.min_confidence = 0.2  # Much lower threshold for better recognition
        self.max_distance = 0.6  # More lenient distance threshold
        
        # Initialize Silent-Face-Anti-Spoofing system
        self.anti_spoof_predictor = None
        self.image_cropper = None
        self.model_dir = None
        self.detection_model_dir = None
        
        if ANTISPOOFING_AVAILABLE:
            try:
                print("🚀 Initializing Silent-Face-Anti-Spoofing system...")
                
                # Set model directory paths
                self.model_dir = os.path.join(silent_face_dir, 'resources', 'anti_spoof_models')
                self.detection_model_dir = os.path.join(silent_face_dir, 'resources', 'detection_model')
                
                # Check if detection model exists
                deploy_file = os.path.join(self.detection_model_dir, 'deploy.prototxt')
                caffemodel_file = os.path.join(self.detection_model_dir, 'Widerface-RetinaFace.caffemodel')
                
                if not os.path.exists(deploy_file) or not os.path.exists(caffemodel_file):
                    print("❌ Detection model files not found")
                    print(f"   Deploy: {deploy_file} - {os.path.exists(deploy_file)}")
                    print(f"   Caffemodel: {caffemodel_file} - {os.path.exists(caffemodel_file)}")
                    self.anti_spoof_predictor = None
                else:
                    # Initialize Silent-Face-Anti-Spoofing components with correct paths
                    # We need to temporarily change the working directory for the model loading
                    original_cwd = os.getcwd()
                    try:
                        # Change to Silent-Face-Anti-Spoofing directory for model loading
                        os.chdir(silent_face_dir)
                        self.anti_spoof_predictor = AntiSpoofPredict(device_id=0)
                        self.image_cropper = CropImage()
                    finally:
                        # Restore original working directory
                        os.chdir(original_cwd)
                    
                    # Check if anti-spoofing models exist
                    if os.path.exists(self.model_dir):
                        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
                        print(f"✅ Loaded {len(model_files)} anti-spoofing models:")
                        for model_file in model_files:
                            print(f"   - {model_file}")
                    else:
                        print("❌ Anti-spoofing models directory not found")
                        self.anti_spoof_predictor = None
                        
                    print("✅ Silent-Face-Anti-Spoofing system initialized successfully")
                
            except Exception as e:
                print(f"❌ Error initializing Silent-Face-Anti-Spoofing: {e}")
                import traceback
                traceback.print_exc()
                self.anti_spoof_predictor = None
        
        print("Initializing web face recognition system...")
        self.load_known_faces_from_db()

    def load_known_faces_from_db(self):
        """Load known faces from database"""
        try:
            print("Connecting to MongoDB...")
            # Try to connect to MongoDB
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            students_collection = db.students
            
            # Test connection
            client.admin.command('ping')
            print("MongoDB connection successful")
            
            # Load students from database
            students = list(students_collection.find({}))
            print(f"Found {len(students)} students in database")
            
            if len(students) == 0:
                print("No students found in database. Please register some students first.")
                client.close()
                return
            
            loaded_count = 0
            for student in students:
                student_id = student.get('usn', '')
                student_name = student.get('name', '')
                
                print(f"Processing student: {student_name} (ID: {student_id})")
                
                # Load face encoding from student data
                if 'face_encoding' in student and student['face_encoding']:
                    try:
                        # Convert list to numpy array
                        encoding = np.array(student['face_encoding'])
                        
                        # Validate encoding shape (should be 128 dimensions)
                        if encoding.shape == (128,):
                            self.known_face_encodings.append(encoding)
                            self.known_face_ids.append(student_id)
                            self.known_face_names.append(student_name)
                            
                            metadata = {
                                'student_id': student_id,
                                'student_name': student_name,
                                'registered_at': student.get('registered_at', ''),
                                'active': student.get('active', True)
                            }
                            self.known_face_metadata.append(metadata)
                            loaded_count += 1
                            print(f"✅ Loaded face encoding for {student_name} (ID: {student_id}) - Shape: {encoding.shape}")
                        else:
                            print(f"❌ Invalid face encoding shape for {student_name}: {encoding.shape}, expected (128,)")
                    except Exception as e:
                        print(f"❌ Error loading face encoding for {student_name}: {e}")
                else:
                    print(f"⚠️ No face encoding found for {student_name} (ID: {student_id})")
            
            print(f"✅ Successfully loaded {loaded_count} face encodings for {len(set(self.known_face_ids))} students")
            client.close()
            
        except Exception as e:
            print(f"❌ Error loading faces from database: {e}")
            print("Face recognition will work with empty database")
            import traceback
            traceback.print_exc()

    def recognize_faces_improved(self, frame):
        """Improved face recognition for web interface with anti-spoofing"""
        print(f"🔍 Starting face recognition with {len(self.known_face_encodings)} known faces")
        
        if len(self.known_face_encodings) == 0:
            print("❌ No known face encodings loaded!")
            return [], [], [], []
        
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                print("❌ Invalid input frame")
                return [], [], [], []
            
            print(f"📷 Processing frame: {frame.shape}")
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print("✅ Frame converted to RGB")
            
            # Detect faces with optimized settings for better detection
            print("🔍 Detecting faces...")
            face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
            print(f"📍 Found {len(face_locations)} face locations")
            
            if not face_locations:
                print("❌ No faces detected in frame")
                return [], [], [], []
            
            # Get face encodings with tolerance parameter
            print("🧠 Extracting face encodings...")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
            print(f"✅ Extracted {len(face_encodings)} face encodings")
            
            if not face_encodings:
                print("❌ Failed to extract face encodings")
                return [], [], [], []
            
            recognized_ids = []
            recognized_names = []
            confidences = []
            
            print("🔍 Identifying faces...")
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                print(f"  Processing face {i+1}/{len(face_encodings)}")
                student_id, student_name, confidence = self._identify_face(face_encoding)
                recognized_ids.append(student_id)
                recognized_names.append(student_name)
                confidences.append(confidence)
                print(f"  Result: {student_name} (ID: {student_id}) - Confidence: {confidence:.3f}")
            
            print(f"✅ Face recognition complete: {len(recognized_ids)} faces processed")
            return face_locations, recognized_ids, recognized_names, confidences
            
        except Exception as e:
            print(f"❌ Error in face recognition: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], [], []

    def check_anti_spoofing(self, frame, face_location):
        """Check if face is real or fake using Silent-Face-Anti-Spoofing"""
        print(f"🛡️ Starting Silent-Face-Anti-Spoofing check...")
        
        if not self.anti_spoof_predictor or not self.image_cropper or not self.model_dir:
            print("⚠️ Silent-Face-Anti-Spoofing not available, assuming real face")
            return True, 0.0  # Assume real if anti-spoofing not available
        
        try:
            # Use Silent-Face-Anti-Spoofing's own face detection
            print("🔍 Using Silent-Face-Anti-Spoofing face detection...")
            bbox = self.anti_spoof_predictor.get_bbox(frame)
            
            if bbox is None:
                print("❌ No face detected by Silent-Face-Anti-Spoofing, assuming real face")
                return True, 0.0
            
            print(f"📍 Silent-Face-Anti-Spoofing bbox: {bbox}")
            
            # Use the helper method for anti-spoofing
            return self._test_anti_spoofing_with_bbox(frame, bbox)
            
        except Exception as e:
            print(f"❌ Error in Silent-Face-Anti-Spoofing detection: {e}")
            import traceback
            traceback.print_exc()
            return True, 0.0  # Assume real on error

    def _test_anti_spoofing_with_bbox(self, frame, bbox):
        """Test anti-spoofing with a given bbox (helper method)"""
        if not self.anti_spoof_predictor or not self.image_cropper or not self.model_dir:
            print("⚠️ Silent-Face-Anti-Spoofing not available, assuming real face")
            return True, 0.0
        
        try:
            print(f"📍 Testing with bbox: {bbox}")
            
            # Validate bbox
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] >= frame.shape[1] or bbox[1] + bbox[3] >= frame.shape[0]:
                print("❌ Invalid bbox, assuming real face")
                return True, 0.0
            
            # Initialize prediction accumulator (same as original test.py)
            prediction = np.zeros((1, 3))
            model_count = 0
            
            # Use all available models for ensemble prediction (exactly like original test.py)
            for model_name in os.listdir(self.model_dir):
                if not model_name.endswith('.pth'):
                    continue
                
                try:
                    print(f"🤖 Processing with model: {model_name}")
                    
                    # Parse the model name to get parameters (same as original)
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    
                    # Prepare image for prediction using CropImage (same as original)
                    param = {
                        "org_img": frame,
                        "bbox": bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    
                    # Handle scale parameter (same as original test.py)
                    if scale is None:
                        param["crop"] = False
                    
                    # Crop and prepare image
                    img = self.image_cropper.crop(**param)
                    print(f"📷 Cropped image shape: {img.shape}")
                    
                    # Get model path and predict (same as original)
                    model_path = os.path.join(self.model_dir, model_name)
                    model_prediction = self.anti_spoof_predictor.predict(img, model_path)
                    
                    # Accumulate predictions (same as original)
                    prediction += model_prediction
                    model_count += 1
                    
                    print(f"📊 Model {model_name} prediction: {model_prediction}")
                    
                except Exception as model_error:
                    print(f"⚠️ Error with model {model_name}: {model_error}")
                    continue
            
            if model_count == 0:
                print("❌ No models processed successfully, assuming real face")
                return True, 0.0
            
            # Get result using the same logic as test_anti_spoofing.py
            label = np.argmax(prediction)
            value = prediction[0][label] / sum(prediction[0])  # Normalize by sum
            
            # Calculate confidence scores for each class
            real_score = prediction[0][1] / sum(prediction[0]) if sum(prediction[0]) > 0 else 0
            fake_score = prediction[0][0] / sum(prediction[0]) if sum(prediction[0]) > 0 else 0
            spoof_score = prediction[0][2] / sum(prediction[0]) if sum(prediction[0]) > 0 else 0
            
            # Use threshold-based detection (like test_anti_spoofing.py)
            threshold = 0.3  # Lower threshold for better real face detection
            is_real = (label == 1 and value > threshold)
            confidence = float(value)
            
            print(f"🛡️ Silent-Face-Anti-Spoofing result:")
            print(f"   - Models used: {model_count}")
            print(f"   - Raw prediction: {prediction[0]}")
            print(f"   - Label: {label} (0=fake, 1=real, 2=spoof)")
            print(f"   - Value: {value:.4f}")
            print(f"   - Real score: {real_score:.4f}")
            print(f"   - Fake score: {fake_score:.4f}")
            print(f"   - Spoof score: {spoof_score:.4f}")
            print(f"   - Threshold: {threshold}")
            print(f"   - Is Real: {is_real}")
            
            if is_real:
                print(f"✅ Image is Real Face. Score: {value:.2f}")
            else:
                print(f"❌ Image is Fake Face. Score: {value:.2f}")
                print(f"   - Detected as: {'Fake' if label == 0 else 'Spoof' if label == 2 else 'Unknown'}")
                print(f"   - Reason: {'Label not 1' if label != 1 else f'Value {value:.3f} <= threshold {threshold}'}")
            
            return bool(is_real), confidence
            
        except Exception as e:
            print(f"❌ Error in Silent-Face-Anti-Spoofing detection: {e}")
            import traceback
            traceback.print_exc()
            return True, 0.0  # Assume real on error

    def _identify_face(self, face_encoding):
        """Identify a face with improved matching algorithm"""
        if len(self.known_face_encodings) == 0:
            print("❌ No known face encodings available for matching")
            return "Unknown", "Unknown", 0.0
        
        try:
            # Validate face encoding
            if face_encoding is None or len(face_encoding) != 128:
                print(f"❌ Invalid face encoding: {face_encoding.shape if hasattr(face_encoding, 'shape') else 'None'}")
                return "Unknown", "Unknown", 0.0
            
            print(f"🔍 Matching face encoding (shape: {face_encoding.shape}) against {len(self.known_face_encodings)} known faces")
            
            # Calculate distances to all known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            print(f"📊 Calculated distances: {face_distances}")
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Use face_recognition's built-in comparison with tolerance
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.recognition_tolerance)
            
            # Calculate confidence score (inverse of distance, normalized)
            confidence = float(max(0, min(1, 1 - (best_distance / 0.6))))  # Convert to Python float
            
            # Debug information
            print(f"🎯 Best match: index={best_match_index}, distance={best_distance:.3f}, confidence={confidence:.3f}")
            print(f"📏 Thresholds: max_distance={self.max_distance}, min_confidence={self.min_confidence}")
            print(f"🔍 Face_recognition match: {matches[best_match_index] if best_match_index < len(matches) else False}")
            
            # Apply thresholds - use face_recognition's built-in comparison
            if best_match_index < len(matches) and matches[best_match_index] and best_distance <= self.max_distance:
                student_id = self.known_face_ids[best_match_index]
                student_name = self.known_face_names[best_match_index]
                print(f"✅ Recognized: {student_name} (ID: {student_id}) with confidence {confidence:.3f}")
                return student_id, student_name, confidence
            else:
                print(f"❌ Face not recognized: distance={best_distance:.3f} > threshold={self.max_distance} or confidence={confidence:.3f} < min={self.min_confidence}")
                return "Unknown", "Unknown", confidence
                
        except Exception as e:
            print(f"❌ Error in face identification: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown", "Unknown", 0.0

    def add_face(self, face_encoding, student_id, student_name):
        """Add a new face to the recognition system"""
        try:
            # Convert to numpy array if it's a list
            if isinstance(face_encoding, list):
                face_encoding = np.array(face_encoding)
            
            # Add to known faces
            self.known_face_encodings.append(face_encoding)
            self.known_face_ids.append(student_id)
            self.known_face_names.append(student_name)
            self.known_face_metadata.append({
                'student_id': student_id,
                'student_name': student_name,
                'added_at': datetime.now()
            })
            
            print(f"Added face for {student_name} (ID: {student_id})")
            return True
            
        except Exception as e:
            print(f"Error adding face: {e}")
            return False

    def reload_faces_from_db(self):
        """Reload all faces from database"""
        print("🔄 Reloading faces from database...")
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.known_face_metadata = []
        self.load_known_faces_from_db()
        print(f"✅ Reloaded {len(self.known_face_encodings)} face encodings")
    
    def test_anti_spoofing_with_image(self, image_path):
        """Test anti-spoofing with a specific image file"""
        if not self.anti_spoof_predictor or not self.image_cropper:
            print("❌ Silent-Face-Anti-Spoofing not available for testing")
            return False, 0.0
        
        try:
            print(f"🧪 Testing anti-spoofing with image: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return False, 0.0
            
            print(f"📷 Loaded image: {image.shape}")
            
            # Get face bounding box using Silent-Face-Anti-Spoofing
            bbox = self.anti_spoof_predictor.get_bbox(image)
            if bbox is None:
                print("❌ No face detected in image")
                return False, 0.0
            
            print(f"📍 Face bbox: {bbox}")
            
            # Test anti-spoofing using the same method as check_anti_spoofing
            is_real, confidence = self._test_anti_spoofing_with_bbox(image, bbox)
            
            print(f"🧪 Test result: {'REAL' if is_real else 'FAKE'} (confidence: {confidence:.3f})")
            return is_real, confidence
            
        except Exception as e:
            print(f"❌ Error testing anti-spoofing: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0

# =============================================================================
# FIXED ATTENDANCE SYSTEM
# =============================================================================

class FixedWebAttendanceSystem:
    """Fixed attendance system for web interface"""
    
    def __init__(self):
        self.face_recognition = FixedWebFaceRecognition()
        print("Fixed web attendance system initialized")

    def recognize_face(self, image):
        """Recognize faces in image"""
        try:
            face_locations, student_ids, student_names, confidences = self.face_recognition.recognize_faces_improved(image)
            return face_locations, student_ids, student_names, confidences
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return [], [], [], []

# =============================================================================
# DATABASE MODELS (SIMPLIFIED)
# =============================================================================

class Student:
    """Student model for database operations"""
    
    @classmethod
    def count(cls):
        """Count total students"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            count = db.students.count_documents({})
            client.close()
            return count
        except:
            return 0

    @classmethod
    def get_department_counts(cls):
        """Get department counts"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            pipeline = [
                {"$group": {"_id": "$department", "count": {"$sum": 1}}}
            ]
            results = list(db.students.aggregate(pipeline))
            client.close()
            
            counts = {}
            for result in results:
                counts[result['_id']] = result['count']
            return counts
        except:
            return {}

class Attendance:
    """Attendance model for database operations"""
    
    @classmethod
    def get_today_count(cls):
        """Get today's attendance count"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            today = datetime.now().strftime('%Y-%m-%d')
            count = db.attendance.count_documents({'date': today})
            client.close()
            return count
        except:
            return 0

    @classmethod
    def from_recognition(cls, student_id, student_name, extra_data=None):
        """Create attendance record from recognition"""
        now = datetime.now()
        return {
            'student_id': student_id,
            'student_name': student_name,
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'timestamp': now,
            'day_of_week': now.strftime('%A'),
            'subject': extra_data.get('subject') if extra_data else 'General',
            'class_name': extra_data.get('class') if extra_data else 'General',
            'branch': extra_data.get('branch', 'Unknown'),
            'sem': extra_data.get('semester', 'Unknown'),
            'section': extra_data.get('section', 'Unknown')
        }

    def save(self):
        """Save attendance record"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            result = db.attendance.insert_one(self)
            client.close()
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving attendance: {e}")
            return None

# =============================================================================
# FLASK WEB APPLICATION
# =============================================================================

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'student_attendance_system_secret_key'

# Initialize compression
compress = Compress()
compress.init_app(app)

# Initialize fixed attendance system
attendance_system = FixedWebAttendanceSystem()

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the dashboard page with statistics"""
    try:
        student_count = Student.count()
        todays_attendance = Attendance.get_today_count()
        dept_counts = Student.get_department_counts()
        
        cse_students = dept_counts.get('CSE', 0)
        ece_students = dept_counts.get('ECE', 0)
        eee_students = dept_counts.get('EEE', 0)
        me_students = dept_counts.get('ME', 0)
        
        # Determine system status
        system_status = "Good"
        status_class = "success"
        
        # Check if face recognition is loaded
        if len(attendance_system.face_recognition.known_face_encodings) == 0:
            system_status = "Warning"
            status_class = "warning"
        
        return render_template('dashboard.html',
                              total_students=student_count,
                              todays_attendance=todays_attendance,
                              cse_students=cse_students,
                              ece_students=ece_students,
                              eee_students=eee_students,
                              me_students=me_students,
                              system_status=system_status,
                              status_class=status_class)
    except Exception as e:
        print(f"Error in dashboard: {e}")
        return render_template('dashboard.html',
                              total_students=0,
                              todays_attendance=0,
                              cse_students=0,
                              ece_students=0,
                              eee_students=0,
                              me_students=0,
                              system_status="Error",
                              status_class="danger")

@app.route('/attendance')
def attendance():
    """Render the attendance taking page"""
    return render_template('attendance.html')

@app.route('/get_attendance_stats')
def get_attendance_stats():
    """Get attendance statistics for the dashboard"""
    try:
        total_students = Student.count()
        todays_attendance = Attendance.get_today_count()
        dept_counts = Student.get_department_counts()
        
        return jsonify({
            'success': True,
            'total_students': total_students,
            'todays_attendance': todays_attendance,
            'department_counts': dept_counts,
            'system_status': 'Active'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    """Process attendance from captured image - FIXED VERSION"""
    try:
        # Check if the request contains JSON data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data provided'}), 400
        
        # Get additional context data from the request if available
        subject = data.get('subject', 'General')
        class_name = data.get('class', 'General')
        session_id = data.get('session_id', None)
        
        # Log the attendance attempt
        logger.info(f"Processing attendance for subject: {subject}, class: {class_name}")
        
        # Get image data
        image_data = data['image']
        
        try:
            # Make sure the image data is properly formatted
            if ',' not in image_data:
                return jsonify({'success': False, 'message': 'Invalid image data format'}), 400
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            if not image_bytes:
                return jsonify({'success': False, 'message': 'Empty image data'}), 400
            
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'success': False, 'message': 'Failed to decode image'}), 400
            
            # Check if image is too small
            if image.shape[0] < 100 or image.shape[1] < 100:
                return jsonify({
                    'success': False, 
                    'message': f'Image too small: {image.shape[1]}x{image.shape[0]}, minimum 100x100 required'
                }), 400
            
            # Resize if image is too large (better performance)
            max_size = 1024
            if image.shape[0] > max_size or image.shape[1] > max_size:
                scale = max_size / max(image.shape[0], image.shape[1])
                image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
                
        except Exception as decode_error:
            logger.error(f"Error decoding image: {decode_error}")
            return jsonify({'success': False, 'error': f"Image decoding error: {str(decode_error)}"}), 400

        # Process with attendance system
        try:
            face_locations, student_ids, student_names, confidences = attendance_system.recognize_face(image)
            print(f"Recognition found {len(face_locations)} faces")
        except Exception as recog_error:
            logger.error(f"Face recognition error: {recog_error}")
            face_locations, student_ids, student_names, confidences = [], [], [], []

        # Track students we've already marked today to prevent duplicates
        today = datetime.now().strftime('%Y-%m-%d')
        processed_students = set()
        
        # Check today's attendance records for already marked students
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            today_records = list(db.attendance.find({'date': today}))
            for record in today_records:
                processed_students.add(record.get('student_id'))
            client.close()
        except Exception as db_error:
            print(f"Database error: {db_error}")
        
        faces = []
        
        for i, (student_id, student_name, confidence) in enumerate(zip(student_ids, student_names, confidences)):
            # Use the same threshold as the face recognition system
            threshold = 0.2
            
            # Check anti-spoofing for each detected face
            is_real = True
            anti_spoof_score = 0.0
            if i < len(face_locations):
                try:
                    is_real, anti_spoof_score = attendance_system.face_recognition.check_anti_spoofing(image, face_locations[i])
                except Exception as spoof_error:
                    print(f"Anti-spoofing error: {spoof_error}")
                    is_real = True  # Assume real on error
                    anti_spoof_score = 0.0

            # Emotion analysis using DeepFace (optional)
            emotion_label = None
            emotion_conf = 0.0
            
            # Try emotion detection if DeepFace is available
            if EMOTION_AVAILABLE and i < len(face_locations):
                try:
                    # face_locations format: (top, right, bottom, left)
                    (top, right, bottom, left) = face_locations[i]
                    
                    # Increase padding for better emotion detection
                    pad = 30  # Increased from 10 to 30
                    h, w = image.shape[:2]
                    t = max(0, top - pad)
                    l = max(0, left - pad)
                    b = min(h, bottom + pad)
                    r = min(w, right + pad)
                    
                    face_crop = image[t:b, l:r]
                    
                    if face_crop.size > 0:
                        # Ensure minimum face size for better emotion detection
                        min_face_size = 100
                        if face_crop.shape[0] < min_face_size or face_crop.shape[1] < min_face_size:
                            # Resize face crop to minimum size while maintaining aspect ratio
                            scale_factor = max(min_face_size / face_crop.shape[0], min_face_size / face_crop.shape[1])
                            new_h = int(face_crop.shape[0] * scale_factor)
                            new_w = int(face_crop.shape[1] * scale_factor)
                            face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Convert BGR to RGB for DeepFace
                        rgb_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        
                        print(f"🔍 Analyzing emotion for face crop: {rgb_face_crop.shape}")
                        
                        # Use multiple models for better emotion detection
                        try:
                            # Try with different backends for better accuracy
                            emo_result = DeepFace.analyze(
                                rgb_face_crop, 
                                actions=['emotion'], 
                                enforce_detection=False,
                                detector_backend='opencv',  # Use OpenCV detector
                                models={'emotion': 'fer2013'}  # Use FER2013 model
                            )
                        except Exception as model_error:
                            print(f"⚠️ Primary emotion model failed, trying fallback: {model_error}")
                            # Fallback to default settings
                            emo_result = DeepFace.analyze(
                                rgb_face_crop, 
                                actions=['emotion'], 
                                enforce_detection=False
                            )
                        
                        # DeepFace may return list or dict depending on version
                        emo = emo_result[0] if isinstance(emo_result, list) else emo_result
                        
                        # Get emotion scores
                        emotion_scores = emo.get('emotion', {})
                        print(f"📊 Raw emotion scores: {emotion_scores}")
                        
                        # Find dominant emotion with better logic
                        if emotion_scores:
                            # Filter out very low confidence emotions
                            filtered_emotions = {k: v for k, v in emotion_scores.items() if v > 5}
                            
                            if filtered_emotions:
                                dominant = max(filtered_emotions, key=filtered_emotions.get)
                                emotion_conf = float(filtered_emotions[dominant]) / 100.0
                                emotion_label = str(dominant)
                                
                                # If confidence is too low, try alternative approach
                                if emotion_conf < 0.3:
                                    # Get the emotion with highest score regardless of threshold
                                    dominant = max(emotion_scores, key=emotion_scores.get)
                                    emotion_conf = float(emotion_scores[dominant]) / 100.0
                                    emotion_label = str(dominant)
                            else:
                                # If all emotions are very low, use the highest one
                                dominant = max(emotion_scores, key=emotion_scores.get)
                                emotion_conf = float(emotion_scores[dominant]) / 100.0
                                emotion_label = str(dominant)
                        
                        print(f"😊 Emotion detected: {emotion_label} ({emotion_conf:.3f})")
                        
                except Exception as emo_err:
                    print(f"❌ Emotion detection error: {emo_err}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try simple emotion detection as fallback
                    try:
                        print("🔄 Trying simple emotion detection fallback...")
                        emotion_label, emotion_conf = detect_emotion_simple(face_crop)
                        print(f"😊 Fallback emotion detected: {emotion_label} ({emotion_conf:.3f})")
                    except Exception as fallback_err:
                        print(f"❌ Fallback emotion detection also failed: {fallback_err}")
                        emotion_label = "neutral"
                        emotion_conf = 0.3
            
            # If DeepFace is not available, try simple emotion detection
            elif not EMOTION_AVAILABLE and i < len(face_locations):
                try:
                    (top, right, bottom, left) = face_locations[i]
                    pad = 30
                    h, w = image.shape[:2]
                    t = max(0, top - pad)
                    l = max(0, left - pad)
                    b = min(h, bottom + pad)
                    r = min(w, right + pad)
                    face_crop = image[t:b, l:r]
                    
                    if face_crop.size > 0:
                        print("🔄 DeepFace not available, using simple emotion detection...")
                        emotion_label, emotion_conf = detect_emotion_simple(face_crop)
                        print(f"😊 Simple emotion detected: {emotion_label} ({emotion_conf:.3f})")
                except Exception as simple_err:
                    print(f"❌ Simple emotion detection failed: {simple_err}")
                    emotion_label = "neutral"
                    emotion_conf = 0.3
            
            # Debug: Print recognition details
            print(f"🎯 Recognition Check: student_id='{student_id}', confidence={confidence:.3f}, threshold={threshold}")
            print(f"🛡️ Anti-spoof Check: is_real={is_real}, anti_spoof_score={anti_spoof_score:.3f}")
            
            # Recognized with sufficient confidence
            if student_id != "Unknown" and confidence > threshold:
                # Check if student already marked attendance today
                already_marked = student_id in processed_students
                print(f"📅 Already marked today: {already_marked}")
                
                # Only mark attendance if face is real AND confidence >= 65%
                anti_spoof_threshold = 0.65
                accepted = bool(is_real and (anti_spoof_score is not None) and (anti_spoof_score >= anti_spoof_threshold))
                print(f"✅ Attendance accepted: {accepted} (is_real={is_real}, anti_spoof_score={anti_spoof_score:.3f}, threshold={anti_spoof_threshold})")
                
                face_result = {
                    'student_id': str(student_id),
                    'name': str(student_name),
                    'confidence': float(confidence),
                    'anti_spoof_score': float(anti_spoof_score),
                    'is_real': bool(is_real),
                    'accepted': bool(accepted),
                    'marked': False,
                    'already_marked': bool(already_marked),
                    'reason': (
                        f"Recognized with confidence {confidence:.2f}"
                        + ("" if is_real else " - Fake face detected")
                        + ("" if accepted else f" - Anti-spoof score {anti_spoof_score:.2f} < {anti_spoof_threshold:.2f}")
                    ),
                    'emotion': (str(emotion_label) if emotion_label else 'Unknown'),
                    'emotion_confidence': float(emotion_conf)
                }
                
                if not already_marked and accepted:
                    # Record new attendance only for real faces
                    try:
                        print(f"📝 Marking attendance for {student_name} (ID: {student_id})")
                        
                        # Get student details from database
                        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
                        db = client.attendance_system
                        student_doc = db.students.find_one({'usn': student_id})
                        client.close()
                        
                        # Prepare extra data with student details
                        extra_data = {
                            'subject': subject,
                            'class': class_name,
                            'branch': student_doc.get('branch', 'Unknown') if student_doc else 'Unknown',
                            'semester': student_doc.get('semester', 'Unknown') if student_doc else 'Unknown',
                            'section': student_doc.get('section', 'Unknown') if student_doc else 'Unknown'
                        }
                        
                        attendance_data = Attendance.from_recognition(student_id, student_name, extra_data)
                        
                        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
                        db = client.attendance_system
                        result = db.attendance.insert_one(attendance_data)
                        client.close()
                        
                        if result.inserted_id:
                            processed_students.add(student_id)
                            face_result['marked'] = True
                            print(f"✅ Successfully marked attendance for {student_name}")
                    except Exception as save_error:
                        print(f"Error saving attendance: {save_error}")
                        face_result['reason'] = f"Database error: {str(save_error)}"
                elif already_marked:
                    face_result['reason'] = "Already marked today"
                    print(f"❌ Attendance NOT marked: Already marked today for {student_name}")
                elif not accepted:
                    if not is_real:
                        face_result['reason'] = "Fake face detected - attendance not marked"
                        print(f"❌ Attendance NOT marked: Fake face detected for {student_name}")
                    else:
                        face_result['reason'] = f"Anti-spoof score below threshold ({anti_spoof_score:.2f} < 0.65) - attendance not marked"
                        print(f"❌ Attendance NOT marked: Low real-face confidence {anti_spoof_score:.2f} for {student_name}")
                
                faces.append(face_result)
            else:
                # Unrecognized face - still check for anti-spoofing
                print(f"❌ Face NOT recognized: student_id='{student_id}', confidence={confidence:.3f} <= threshold={threshold}")
                is_real_unknown = True
                anti_spoof_score_unknown = 0.0
                if i < len(face_locations):
                    try:
                        is_real_unknown, anti_spoof_score_unknown = attendance_system.face_recognition.check_anti_spoofing(image, face_locations[i])
                    except Exception as spoof_error:
                        print(f"Anti-spoofing error for unknown face: {spoof_error}")
                        is_real_unknown = True
                        anti_spoof_score_unknown = 0.0
                
                faces.append({
                    'student_id': 'Unknown',
                    'name': 'Unknown Person',
                    'confidence': float(confidence),
                    'anti_spoof_score': float(anti_spoof_score_unknown),
                    'is_real': bool(is_real_unknown),
                    'marked': False,
                    'reason': ("Confidence too low" if confidence > 0.2 else "No match found") + ("" if is_real_unknown else " - Fake face detected"),
                    'emotion': (str(emotion_label) if emotion_label else 'Unknown'),
                    'emotion_confidence': float(emotion_conf)
                })

        return jsonify({
            'success': True,
            'faces': faces,
            'total_faces': int(len(face_locations))
        })

    except Exception as e:
        logger.error(f"Error in process_attendance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/registration', methods=['GET', 'POST'])
def registration():
    """Handle student registration"""
    if request.method == 'POST':
        try:
            logger.info('Received POST to /registration')
            # Get form data
            student_info = {
                'usn': request.form['usn'],
                'name': request.form['name'],
                'semester': request.form['semester'],
                'branch': request.form['branch'],
                'section': request.form['section'],
                'phone': request.form.get('phone'),
                'address': request.form.get('address')
            }
            logger.info(f"Registration form data: usn={student_info.get('usn')}, name={student_info.get('name')}")

            # Process uploaded image (optional)
            photo_key = 'photo_0'
            if photo_key in request.form and request.form.get(photo_key):
                try:
                    # Decode base64 image
                    image_data = request.form.get(photo_key)
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    
                    # Convert to numpy array
                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Convert to RGB for face_recognition
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Find face locations
                        face_locations = face_recognition.face_locations(rgb_image)
                        
                        if face_locations:
                            # Get face encodings
                            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                            
                            if face_encodings:
                                # Use the first face encoding
                                face_encoding = face_encodings[0]
                                student_info['face_encoding'] = face_encoding.tolist()
                                logger.info("Face encoding extracted successfully")
                            else:
                                logger.warning("No face encodings found in image")
                        else:
                            logger.warning("No faces detected in image")
                    else:
                        logger.warning("Could not decode image")
                        
                except Exception as img_error:
                    logger.error(f"Error processing image: {img_error}")
            else:
                logger.info("No photo provided; registering student without face encoding")

            # Save to MongoDB
            try:
                client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
                db = client.attendance_system
                
                # Create student document
                student_doc = {
                    'usn': student_info['usn'],
                    'name': student_info['name'],
                    'semester': student_info['semester'],
                    'branch': student_info['branch'],
                    'section': student_info['section'],
                    'phone': student_info.get('phone'),
                    'address': student_info.get('address'),
                    'registered_at': datetime.now(),
                    'face_encoding': student_info.get('face_encoding'),
                    'active': True
                }
                
                # Insert into database
                result = db.students.insert_one(student_doc)
                client.close()
                
                if result.inserted_id:
                    logger.info(f'Student saved to DB with id: {result.inserted_id}')
                    
                    # Update face recognition system if face encoding exists
                    if 'face_encoding' in student_info:
                        attendance_system.face_recognition.add_face(
                            student_info['face_encoding'],
                            student_info['usn'],
                            student_info['name']
                        )
                        flash(f"{student_info['name']} registered successfully with face recognition enabled.", 'success')
                    else:
                        flash(f"{student_info['name']} registered successfully (no face photo provided).", 'success')
                    
                    return redirect(url_for('registration'))
                else:
                    flash("Failed to save student to database", 'danger')
                    
            except Exception as db_error:
                logger.error(f"Database error: {db_error}")
                flash(f"Database error: {str(db_error)}", 'danger')
                
        except Exception as e:
            logger.error(f"Error in registration: {e}")
            flash(f"Error during registration: {str(e)}", 'danger')

    return render_template('registration.html')

@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    """Reload faces from database"""
    try:
        attendance_system.face_recognition.reload_faces_from_db()
        return jsonify({
            'success': True,
            'message': f'Successfully reloaded {len(attendance_system.face_recognition.known_face_encodings)} face encodings'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error reloading faces: {str(e)}'}), 500

@app.route('/analytics')
def analytics():
    """Serve the analytics page"""
    today_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('analytics.html', today_date=today_date)

@app.route('/get_attendance')
def get_attendance():
    """Get attendance records with filtering"""
    try:
        # Get filter parameters
        date_filter = request.args.get('date')
        branch_filter = request.args.get('branch')
        semester_filter = request.args.get('semester')
        section_filter = request.args.get('section')
        
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        
        # Build query
        query = {}
        if date_filter:
            query['date'] = date_filter
        if branch_filter:
            query['branch'] = branch_filter
        if semester_filter:
            query['sem'] = semester_filter
        if section_filter:
            query['section'] = section_filter
        
        # Get attendance records
        attendance_records = list(db.attendance.find(query).sort('date', -1).sort('time', -1))
        
        # Get all students to show absent ones
        students_query = {}
        if branch_filter:
            students_query['branch'] = branch_filter
        if semester_filter:
            students_query['semester'] = semester_filter  # Fixed: use 'semester' field from students collection
        if section_filter:
            students_query['section'] = section_filter
            
        all_students = list(db.students.find(students_query))
        
        # Create a comprehensive attendance list
        attendance_data = []
        
        # Add present students
        for record in attendance_records:
            attendance_data.append({
                'student_id': record.get('student_id', 'Unknown'),
                'student_name': record.get('student_name', 'Unknown'),
                'branch': record.get('branch', 'Unknown'),
                'sem': record.get('sem', 'Unknown'),
                'section': record.get('section', 'Unknown'),
                'date': record.get('date', 'Unknown'),
                'time': record.get('time', 'Unknown'),
                'status': 'Present'
            })
        
        # Add absent students (students not in attendance records for the date)
        if date_filter:
            present_student_ids = {record.get('student_id') for record in attendance_records}
            for student in all_students:
                if student.get('usn') not in present_student_ids:
                    attendance_data.append({
                        'student_id': student.get('usn', 'Unknown'),
                        'student_name': student.get('name', 'Unknown'),
                        'branch': student.get('branch', 'Unknown'),
                        'sem': student.get('semester', 'Unknown'),  # Fixed: use 'semester' from students collection
                        'section': student.get('section', 'Unknown'),
                        'date': date_filter,
                        'time': '',
                        'status': 'Absent'
                    })
        
        client.close()
        
        return jsonify(attendance_data)
        
    except Exception as e:
        logger.error(f"Error getting attendance: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("Starting Fixed Integrated Student Attendance System...")
    print(f"Face recognition loaded: {len(attendance_system.face_recognition.known_face_encodings)} faces")
    
    # Add test page route
    @app.route('/test_anti_spoofing_page')
    def test_anti_spoofing_page():
        """Serve the anti-spoofing test page"""
        return render_template('test_anti_spoofing.html')
    
    # Add test route for anti-spoofing
    @app.route('/test_anti_spoofing', methods=['POST'])
    def test_anti_spoofing():
        """Test anti-spoofing with uploaded image"""
        try:
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No image selected'}), 400
            
            # Save uploaded image temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Test anti-spoofing
            is_real, confidence = attendance_system.face_recognition.test_anti_spoofing_with_image(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return jsonify({
                'success': True,
                'is_real': is_real,
                'confidence': confidence,
                'result': 'REAL FACE' if is_real else 'FAKE FACE'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Automatically open the web browser
    threading.Timer(1.0, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    
    # Start Flask app
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        import traceback
        traceback.print_exc()
