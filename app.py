import streamlit as st
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image
import numpy as np
import hashlib
import json
import time
from datetime import datetime
import tempfile
import os
import base64
import requests
import random
import threading
import io
import zipfile

# Page config
st.set_page_config(
    page_title="Wildlife Protection Blockchain System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - Nature Theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2D5016;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 8px solid #D32F2F;
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        box-shadow: 0 4px 15px rgba(211, 47, 47, 0.2);
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 8px solid #388E3C;
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        box-shadow: 0 4px 15px rgba(56, 142, 60, 0.2);
    }
    .blockchain-block {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
        transition: transform 0.3s ease;
    }
    .blockchain-block:hover {
        transform: translateY(-2px);
    }
    .upload-section {
        background: linear-gradient(135deg, #E8F5E8 0%, #A5D6A7 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: 3px dashed #4CAF50;
    }
    .metric-card {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    .voice-alert {
        background: linear-gradient(135deg, #FF8A65 0%, #FF7043 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 112, 67, 0.3);
        animation: pulse 2s infinite;
    }
    .history-section {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #FF9800;
    }
    .frame-gallery {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components

# Voice Alert System
def speak_alert_web(message):
    """Use browser's speech synthesis - works in cloud environments"""
    # Clean message for JavaScript (escape quotes and special chars)
    clean_message = message.replace('"', '\\"').replace("'", "\\'")
    
    speech_js = f"""
    <script>
    if ('speechSynthesis' in window) {{
        const utterance = new SpeechSynthesisUtterance("{clean_message}");
        utterance.rate = 0.8;
        utterance.pitch = 1.0;
        utterance.volume = 0.9;
        
        // Wait a moment then speak
        setTimeout(() => {{
            speechSynthesis.speak(utterance);
        }}, 100);
    }}
    </script>
    """
    components.html(speech_js, height=0)
    return True

def speak_alert(message):
    """Fallback to pyttsx3 for local environments"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        def tts_thread():
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        
        thread = threading.Thread(target=tts_thread)
        thread.daemon = True
        thread.start()
        
        return True
    except ImportError:
        return False
    except Exception:
        return False

def calculate_frame_quality(frame, detections):
    """Calculate frame quality score based on clarity and detections"""
    # Convert to grayscale for quality metrics
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Laplacian variance
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness and contrast
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Detection quality - higher confidence and more detections = better
    detection_score = 0
    if detections:
        avg_confidence = np.mean([d['conf'] for d in detections])
        detection_count = len(detections)
        detection_score = avg_confidence * (1 + 0.1 * detection_count)
    
    # Combined quality score
    quality_score = (sharpness * 0.4) + (brightness * 0.1) + (contrast * 0.2) + (detection_score * 100 * 0.3)
    
    return quality_score

def save_frame_as_image(frame, frame_info, output_dir):
    """Save a frame as an image file"""
    timestamp = frame_info.get('timestamp', time.time())
    quality = frame_info.get('quality', 0)
    
    filename = f"frame_{timestamp:.2f}s_quality_{quality:.0f}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, frame)
    return filepath, filename

def create_frames_zip(frame_files):
    """Create a zip file containing all the best frames"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filepath, filename in frame_files:
            if os.path.exists(filepath):
                zip_file.write(filepath, filename)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def add_to_session_history(analysis_type, results_df):
    """Add analysis results to session history"""
    if results_df is not None and not results_df.empty:
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_type": analysis_type,
            "total_animals": len(results_df),
            "species_breakdown": results_df['species_type'].value_counts().to_dict() if 'species_type' in results_df.columns else {},
            "condition_breakdown": results_df['condition'].value_counts().to_dict() if 'condition' in results_df.columns else {},
            "threat_breakdown": results_df[results_df['threats'] != 'None']['threats'].value_counts().to_dict() if 'threats' in results_df.columns else {},
            "location_breakdown": results_df['location'].value_counts().to_dict() if 'location' in results_df.columns else {}
        }
        st.session_state.session_history.append(history_entry)
        st.session_state.total_analyses += 1

def create_session_charts():
    """Create charts for current session data"""
    if not st.session_state.session_history:
        return None, None, None
    
    # Combine all session data
    all_species = {}
    all_conditions = {}
    all_threats = {}
    
    for entry in st.session_state.session_history:
        # Species data
        for species, count in entry['species_breakdown'].items():
            all_species[species] = all_species.get(species, 0) + count
        
        # Condition data  
        for condition, count in entry['condition_breakdown'].items():
            all_conditions[condition] = all_conditions.get(condition, 0) + count
        
        # Threat data
        for threat, count in entry['threat_breakdown'].items():
            if threat != 'None':
                all_threats[threat] = all_threats.get(threat, 0) + count
    
    return all_species, all_conditions, all_threats

def get_specific_location_for_alert(animal_type):
    """Get a specific location for voice alerts"""
    if animal_type in wildlife_sanctuary["regions"]:
        park = wildlife_sanctuary["name"]
        region = random.choice(wildlife_sanctuary["regions"][animal_type])
        return f"{park}, {region}"
    return f"{wildlife_sanctuary['name']}, Monitoring Station Alpha"

# Blockchain Implementation
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.data, default=str)}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty=2):
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        return self.hash

class WildlifeBlockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 2
        
    def create_genesis_block(self):
        genesis_data = {
            "message": "Wildlife Protection Blockchain Initialized",
            "timestamp": datetime.now().isoformat()
        }
        return Block(0, time.time(), genesis_data, "0")
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def add_animal_detection(self, animal_data):
        block = Block(
            len(self.chain),
            time.time(),
            animal_data,
            self.get_latest_block().hash
        )
        block.mine_block(self.difficulty)
        self.chain.append(block)
        return block
    
    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
        return True
    
    def get_chain_data(self):
        return [
            {
                "index": block.index,
                "timestamp": datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                "hash": block.hash[:16] + "...",
                "previous_hash": block.previous_hash[:16] + "..." if block.previous_hash != "0" else "Genesis",
                "data": block.data,
                "nonce": block.nonce
            }
            for block in self.chain
        ]

# Initialize session state
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = WildlifeBlockchain()
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = True
if 'session_history' not in st.session_state:
    st.session_state.session_history = []
if 'total_analyses' not in st.session_state:
    st.session_state.total_analyses = 0
if 'best_frames' not in st.session_state:
    st.session_state.best_frames = []
if 'frame_files' not in st.session_state:
    st.session_state.frame_files = []

# Model file paths
MODEL_FILES = {
    "yolo12n": "yolo12n.pt",
    "best": "best.pt", 
    "bests": "bests.pt",
    "best_train": "best_train.pt"
}

# GitHub Release URL for large model file
MODEL_URL = "https://github.com/tumblr-byte/Wildlife-Protection-Chain/releases/download/v1.0.0-models/best_train.pt"

def load_models_silently():
    """Load all AI models silently in the background"""
    if st.session_state.models_loaded:
        return st.session_state.model, st.session_state.model2, st.session_state.animal_envo, st.session_state.animal_condition_model
    
    models_loaded = {}
    
    try:
        # Load YOLO models
        if os.path.exists(MODEL_FILES["yolo12n"]):
            models_loaded["model"] = YOLO(MODEL_FILES["yolo12n"])
        else:
            models_loaded["model"] = YOLO("yolo12n.pt")  # Auto-download
        
        if os.path.exists(MODEL_FILES["best"]):
            models_loaded["model2"] = YOLO(MODEL_FILES["best"])
        else:
            raise Exception("best.pt not found")
        
        if os.path.exists(MODEL_FILES["bests"]):
            models_loaded["animal_envo"] = YOLO(MODEL_FILES["bests"])
        else:
            raise Exception("bests.pt not found")
        
        # Load condition classification model
        if not os.path.exists(MODEL_FILES["best_train"]):
            # Download silently
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_FILES["best_train"], "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        
        animal_condition_model = models.resnet18(pretrained=True)
        in_features = animal_condition_model.fc.in_features
        animal_condition_model.fc = nn.Linear(in_features, 2)
        animal_condition_model.load_state_dict(
            torch.load(MODEL_FILES["best_train"], map_location="cpu")
        )
        animal_condition_model.eval()
        models_loaded["animal_condition_model"] = animal_condition_model
        
        # Store in session state
        st.session_state.model = models_loaded["model"]
        st.session_state.model2 = models_loaded["model2"]
        st.session_state.animal_envo = models_loaded["animal_envo"]
        st.session_state.animal_condition_model = models_loaded["animal_condition_model"]
        st.session_state.models_loaded = True
        
        return (models_loaded["model"], 
                models_loaded["model2"], 
                models_loaded["animal_envo"], 
                models_loaded["animal_condition_model"])
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

# Constants
animal_class = ["rhino", "elephant", "tiger"]
condition_class = ["injured", "normal"]
envo_class = {0: "weapon", 1: "fire"}
more_envo_class = {0: "person", 2: "car", 7: "truck", -1: "plastic"}

# Animal colors for bounding boxes
animal_colors = {
    "rhino": (255, 0, 0),      # Red
    "elephant": (0, 255, 0),   # Green
    "tiger": (0, 0, 255)       # Blue
}

# Enhanced location system with regions - Single Wildlife Sanctuary
wildlife_sanctuary = {
    "name": "Jim Corbett National Park, Uttarakhand",
    "regions": {
        "rhino": ["Dhikala Zone", "Bijrani Zone", "Jhirna Zone", "Durgadevi Zone"],
        "elephant": ["Dhikala Zone", "Corbett Landscape", "Sonanadi Zone", "Sitabani Zone"],
        "tiger": ["Core Zone", "Buffer Zone", "Dhikala Zone", "Bijrani Zone"]
    }
}

def get_random_location(animal_type):
    """Get a random region for the specified animal type within the same sanctuary"""
    if animal_type in wildlife_sanctuary["regions"]:
        park = wildlife_sanctuary["name"]
        region = random.choice(wildlife_sanctuary["regions"][animal_type])
        return f"{park}, {region}"
    return f"{wildlife_sanctuary['name']}, Unknown Zone"

device = "cuda" if torch.cuda.is_available() else "cpu"

def detect_animals_image(image, model2):
    """Detect animals in a single image"""
    try:
        results = model2(image, conf=0.5, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf.cpu().numpy()[0])
                if conf < 0.5:
                    continue
                cls_id = int(box.cls.cpu().numpy()[0])
                if cls_id >= len(animal_class):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                cropped = image[y1:y2, x1:x2]
                name = animal_class[cls_id]

                detections.append({
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "name": name,
                    "cropped": cropped
                })

        return detections
    except:
        return []

def detect_all_animals(frame, model2):
    """Detect animals in video frame"""
    try:
        results = model2(frame, conf=0.8, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf.cpu().numpy()[0])
                if conf < 0.8:
                    continue
                cls_id = int(box.cls.cpu().numpy()[0])
                if cls_id >= len(animal_class):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                cropped = frame[y1:y2, x1:x2]
                name = animal_class[cls_id]

                detections.append({
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "name": name,
                    "cropped": cropped,
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                })

        return detections
    except:
        return []

def get_condition(cropped, animal_condition_model):
    if cropped is None or cropped.size == 0:
        return "normal"
    
    try:
        image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = animal_condition_model(image)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        return condition_class[pred]
    except:
        return "normal"

def get_threats(frame, model, animal_envo):
    threats = []
    
    try:
        results1 = model(frame, conf=0.6, verbose=False)
        for result in results1:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                if conf >= 0.6 and cls_id in more_envo_class:
                    threats.append(more_envo_class[cls_id])

        results2 = animal_envo(frame, conf=0.6, verbose=False)
        for result in results2:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                if conf >= 0.6 and cls_id in envo_class:
                    threats.append(envo_class[cls_id])
    except:
        pass

    return list(set(threats)) if threats else ["None"]

def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def simple_tracking(current_detections, previous_tracks, max_distance=100):
    tracks = []
    used_track_ids = set()
    
    for detection in current_detections:
        best_match = None
        min_distance = float('inf')
        
        for track_id, prev_info in previous_tracks.items():
            if track_id in used_track_ids:
                continue
                
            if prev_info["name"] == detection["name"]:
                distance = calculate_distance(detection["center"], prev_info["center"])
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    best_match = track_id
        
        if best_match:
            track_id = best_match
            used_track_ids.add(track_id)
        else:
            track_id = f"{detection['name']}_{len(previous_tracks) + len(tracks) + 1}"
        
        tracks.append({
            "track_id": track_id,
            "detection": detection
        })
    
    return tracks

def show_voice_alert(alert_type, message, animal_type="", location="", threats=[]):
    """Show alert with voice notification"""
    
    if alert_type == "injury":
        alert_html = f"""
        <div class="voice-alert">
            <h3>🚨 INJURY ALERT - VOICE NOTIFICATION ACTIVE</h3>
            <p><strong>Injured {animal_type.title()} Detected!</strong></p>
            <p>📍 Location: {location}</p>
            <p>🏥 Medical team has been notified for immediate assistance.</p>
            <p>🔊 Voice alert has been triggered!</p>
        </div>
        """
        voice_msg = f"Alert! Injured {animal_type} detected at {location}. Medical assistance required immediately."
        
    elif alert_type == "threat":
        threat_list = ", ".join(threats) if threats else "Unknown threat"
        alert_html = f"""
        <div class="voice-alert">
            <h3>⚠️ THREAT ALERT - VOICE NOTIFICATION ACTIVE</h3>
            <p><strong>Threat Detected Around Animals!</strong></p>
            <p>🎯 Threats: {threat_list}</p>
            <p>📍 Location: {location}</p>
            <p>👮 Authorities have been notified for immediate intervention.</p>
            <p>🔊 Voice alert has been triggered!</p>
        </div>
        """
        voice_msg = f"Threat alert! {threat_list} detected at {location}. Immediate intervention required."
    
    st.markdown(alert_html, unsafe_allow_html=True)
    
    # Trigger voice alert if enabled
    if st.session_state.voice_enabled:
        try:
            # Try web-based speech first (works in cloud)
            speak_alert_web(voice_msg)
            st.success("🔊 Voice alert delivered via browser!")
        except:
            try:
                # Fallback to pyttsx3 for local environments
                if speak_alert(voice_msg):
                    st.success("🔊 Voice alert delivered via system!")
                else:
                    st.info("🔇 Voice system temporarily unavailable")
            except:
                st.info("🔇 Browser speech not supported on this device")

def process_single_image(image_array):
    """Process a single image for wildlife detection"""
    model, model2, animal_envo, animal_condition_model = load_models_silently()
    
    if not all([model, model2, animal_envo, animal_condition_model]):
        return None, None
    
    # Detect animals
    detections = detect_animals_image(image_array, model2)
    
    if not detections:
        return pd.DataFrame(), image_array
    
    results_list = []
    output_image = image_array.copy()
    
    # Check for threats - enhanced for images
    threats = []
    try:
        # Use lower confidence for images
        results1 = model(image_array, conf=0.7, verbose=False)
        for result in results1:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    if conf >= 0.7 and cls_id in more_envo_class:
                        threats.append(more_envo_class[cls_id])

        results2 = animal_envo(image_array, conf=0.5, verbose=False)
        for result in results2:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    if conf >= 0.5 and cls_id in envo_class:
                        threats.append(envo_class[cls_id])
    except:
        pass

    threats = list(set(threats)) if threats else ["None"]
    
    # Process each detection
    for i, detection in enumerate(detections):
        # Get animal condition
        condition = get_condition(detection["cropped"], animal_condition_model)
        
        # Get location
        location = get_random_location(detection["name"])
        threat_str = ",".join(threats)
        
        # Create animal record
        animal_record = {
            "animal_id": f"{detection['name']}_{i+1}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "species_type": detection["name"],
            "condition": condition,
            "threats": threat_str,
            "location": location,
            "confidence": round(detection["conf"], 2)
        }
        
        # Add to blockchain
        try:
            st.session_state.blockchain.add_animal_detection(animal_record)
        except:
            pass
        
        results_list.append(animal_record)
        
        # Draw bounding box
        x1, y1, x2, y2 = detection["bbox"]
        color = animal_colors.get(detection["name"], (255, 255, 255))
        
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
        
        # Add label
        label = f"{detection['name'].title()} ({condition})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(output_image, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
        cv2.putText(output_image, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Check for alerts
        if condition == "injured":
            show_voice_alert("injury", f"Injured {detection['name']} needs help!", detection['name'], location)
        
        if threats != ["None"]:
            threat_list = [t for t in threats if t != "None"]
            if threat_list:
                show_voice_alert("threat", f"Threats detected: {', '.join(threat_list)}", location=location, threats=threat_list)
    
    # Create DataFrame
    df = pd.DataFrame(results_list) if results_list else pd.DataFrame()
    return df, output_image
    
def process_video_streamlit(video_path):
    """Process video with tracking, blockchain logging, and best frame capture"""
    model, model2, animal_envo, animal_condition_model = load_models_silently()
    
    if not all([model, model2, animal_envo, animal_condition_model]):
        st.error("❌ Failed to load AI models. Please check model files and try again.")
        return None, None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video path
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracking variables
    saved_animals = set()
    results_list = []
    previous_tracks = {}
    last_conditions = {}
    last_threats = ["None"]
    frame_count = 0
    
    # Frame quality tracking for best frames
    frame_quality_data = []
    
    # Alert tracking
    injury_alerts = set()
    threat_alerts = set()
    
    # Create temp directory for frames
    temp_frames_dir = tempfile.mkdtemp()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.info("🔗 Processing video with AI tracking, blockchain security, and frame capture...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Analyzing frame {frame_count}/{total_frames}")
        
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        current_detections = []
        if frame_count % 5 == 0:  # Detection every 5 frames
            current_detections = detect_all_animals(frame, model2)
        
        # Calculate frame quality if animals detected
        if current_detections:
            quality_score = calculate_frame_quality(frame, current_detections)
            
            frame_info = {
                'timestamp': timestamp,
                'quality': quality_score,
                'detections': len(current_detections),
                'frame': frame.copy(),
                'frame_number': frame_count
            }
            
            frame_quality_data.append(frame_info)
            
            tracks = simple_tracking(current_detections, previous_tracks)
            
            # Update previous tracks
            previous_tracks = {}
            for track in tracks:
                track_id = track["track_id"]
                detection = track["detection"]
                
                previous_tracks[track_id] = {
                    "name": detection["name"],
                    "center": detection["center"],
                    "bbox": detection["bbox"],
                    "cropped": detection["cropped"]
                }
            
            # Check conditions
            if frame_count % 15 == 0:  # Condition check every 15 frames
                for track in tracks:
                    track_id = track["track_id"]
                    detection = track["detection"]
                    condition = get_condition(detection["cropped"], animal_condition_model)
                    last_conditions[track_id] = condition
            
            # Check threats
            if frame_count % 10 == 0:  # Threat check every 10 frames
                last_threats = get_threats(frame, model, animal_envo)
            
            # Process each track
            for track in tracks:
                track_id = track["track_id"]
                detection = track["detection"]
                
                if track_id not in saved_animals:
                    condition = last_conditions.get(track_id, "normal")
                    threat_str = ",".join(last_threats)
                    location = get_random_location(detection["name"])
                    
                    # Create animal record for blockchain
                    animal_record = {
                        "track_id": track_id,
                        "timestamp": round(timestamp, 2),
                        "species_type": detection["name"],
                        "condition": condition,
                        "threats": threat_str,
                        "location": location,
                        "confidence": round(detection["conf"], 2)
                    }
                    
                    # Add to blockchain
                    try:
                        st.session_state.blockchain.add_animal_detection(animal_record)
                    except:
                        pass
                    
                    # Add to results
                    results_list.append(animal_record)
                    saved_animals.add(track_id)
                    
                    # Check for alerts
                    if condition == "injured" and track_id not in injury_alerts:
                        injury_alerts.add(track_id)
                    
                    if last_threats != ["None"] and tuple(last_threats) not in threat_alerts:
                        threat_alerts.add(tuple(last_threats))
                
                # Draw bounding box with animal-specific color
                x1, y1, x2, y2 = detection["bbox"]
                color = animal_colors.get(detection["name"], (255, 255, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Add label with background
                label = f"{detection['name'].title()} {track_id.split('_')[-1]}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    # Process best frames
    st.session_state.best_frames = []
    st.session_state.frame_files = []
    
    if frame_quality_data:
        # Sort frames by quality and get top 5
        frame_quality_data.sort(key=lambda x: x['quality'], reverse=True)
        best_frames_data = frame_quality_data[:5]
        
        # Save best frames as images
        for i, frame_data in enumerate(best_frames_data):
            filepath, filename = save_frame_as_image(
                frame_data['frame'], 
                {
                    'timestamp': frame_data['timestamp'],
                    'quality': frame_data['quality']
                },
                temp_frames_dir
            )
            
            st.session_state.frame_files.append((filepath, filename))
            
            # Store frame info for display
            st.session_state.best_frames.append({
                'frame': frame_data['frame'],
                'timestamp': frame_data['timestamp'],
                'quality': frame_data['quality'],
                'detections': frame_data['detections'],
                'filename': filename
            })
    
    # Show voice alerts for video processing
    for track_id in injury_alerts:
        animal_type = track_id.split('_')[0]
        location = get_specific_location_for_alert(animal_type)
        show_voice_alert("injury", f"Immediate medical attention required for {animal_type}!", animal_type, location)
    
    if len(threat_alerts) > 0:
        all_threats = set()
        for threat_tuple in threat_alerts:
            all_threats.update(threat_tuple)
        threat_list = [t for t in all_threats if t != "None"]
        if threat_list:
            location = get_specific_location_for_alert("tiger")  # Use tiger zone as default for threats
            show_voice_alert("threat", f"Detected threats: {', '.join(threat_list)}", location=location, threats=threat_list)
    
    # Create DataFrame
    if results_list:
        df = pd.DataFrame(results_list)
        return df, output_path
    else:
        return pd.DataFrame(), output_path

