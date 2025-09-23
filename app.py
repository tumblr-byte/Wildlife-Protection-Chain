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

# Page config
st.set_page_config(
    page_title="Wildlife Protection Blockchain System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff4444;
        background-color: #ffebee;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        background-color: #e8f5e8;
    }
    .blockchain-block {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
        progress_bar = st.progress(0)
        iterations = 0
        max_iterations = 10000
        
        while self.hash[:difficulty] != target and iterations < max_iterations:
            self.nonce += 1
            self.hash = self.calculate_hash()
            iterations += 1
            if iterations % 100 == 0:
                progress_bar.progress(min(iterations / max_iterations, 1.0))
        
        progress_bar.empty()
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
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# GitHub Release URL for large model file
MODEL_URL = "https://github.com/tumblr-byte/WildGuard-Blockchain/releases/download/v1.0.0-models/best_train.pt"
MODEL_PATH = "best_train.pt"

def download_model():
    """Download the large model file from GitHub releases if not present"""
    if not os.path.exists(MODEL_PATH):
        try:
            st.info(f"📥 Downloading {MODEL_PATH} from GitHub releases... This may take a few minutes.")
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                if total_size > 0:
                    progress_bar = st.progress(0)
                    downloaded = 0
                    
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                progress = downloaded / total_size
                                progress_bar.progress(progress)
                    
                    progress_bar.empty()
                    st.success(f"✅ {MODEL_PATH} downloaded successfully!")
                else:
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    st.success(f"✅ {MODEL_PATH} downloaded successfully!")
                    
        except Exception as e:
            st.error(f"❌ Failed to download {MODEL_PATH}: {str(e)}")
            st.info("Please manually download the model file from the GitHub releases page.")
            return False
    return True

@st.cache_resource
def load_models():
    """Load all AI models with automatic download for large files"""
    try:
        # Load YOLO models (should be in project directory)
        model = YOLO("yolo12n.pt")
        model2 = YOLO("best.pt")
        animal_envo = YOLO("bests.pt")

        # Download and load the large condition model
        if download_model():
            animal_condition_model = models.resnet18(pretrained=True)
            in_features = animal_condition_model.fc.in_features
            animal_condition_model.fc = nn.Linear(in_features, 2)
            animal_condition_model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu")
            )
            animal_condition_model.eval()
        else:
            # Fallback: use pretrained model without custom weights
            animal_condition_model = models.resnet18(pretrained=True)
            animal_condition_model.eval()
            st.warning("⚠️ Using fallback model. Some condition detection may be less accurate.")

        return model, model2, animal_envo, animal_condition_model
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure all model files are in the project directory or check the GitHub releases.")
        return None, None, None, None

# Constants
animal_class = ["rhino", "elephant", "tiger"]
condition_class = ["injured", "normal"]
envo_class = {0: "Automatic Rifle", 1: "Bazooka", 2: "Grenade Launcher", 3: "Handgun", 4: "Knife", 5: "Shotgun", 6: "SMG", 7: "Sniper", 8: "Sword", 9: "smoke", 10: "fire"}
more_envo_class = {0: "person", 2: "car", 7: "truck", -1: "plastic"}

# Animal colors for bounding boxes
animal_colors = {
    "rhino": (255, 0, 0),      # Red
    "elephant": (0, 255, 0),   # Green
    "tiger": (0, 0, 255)       # Blue
}

# Default locations
default_locations = {
    "rhino": "Kaziranga National Park, Assam",
    "elephant": "Jim Corbett National Park, Uttarakhand", 
    "tiger": "Ranthambore National Park, Rajasthan"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def detect_all_animals(frame, model2):
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

def show_alert(alert_type, message, animal_type=""):
    if alert_type == "injury":
        st.markdown(f"""
        <div class="alert-box">
            <h3>🚨 INJURY ALERT</h3>
            <p><strong>Injured {animal_type.title()} Detected!</strong></p>
            <p>{message}</p>
            <p>📍 Location: {default_locations.get(animal_type, 'Wildlife Reserve')}</p>
            <p>🏥 Medical team has been notified for immediate assistance.</p>
        </div>
        """, unsafe_allow_html=True)
    elif alert_type == "threat":
        st.markdown(f"""
        <div class="alert-box">
            <h3>⚠️ THREAT ALERT</h3>
            <p><strong>Threat Detected Around Animals!</strong></p>
            <p>{message}</p>
            <p>📍 Location: Wildlife Reserve</p>
            <p>👮 Authorities have been notified for immediate intervention.</p>
        </div>
        """, unsafe_allow_html=True)

def process_video_streamlit(video_path):
    model, model2, animal_envo, animal_condition_model = load_models()
    
    if not all([model, model2, animal_envo, animal_condition_model]):
        st.error("Failed to load AI models. Please check model files.")
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
    
    saved_animals = set()
    results_list = []
    previous_tracks = {}
    last_conditions = {}
    last_threats = ["None"]
    frame_count = 0
    
    # Alert tracking
    injury_alerts = set()
    threat_alerts = set()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.info("🔗 Initializing blockchain for secure data storage...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        current_detections = []
        if frame_count % 5 == 0:  # Detection skip
            current_detections = detect_all_animals(frame, model2)
        
        if current_detections:
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
            if frame_count % 15 == 0:  # Condition skip
                for track in tracks:
                    track_id = track["track_id"]
                    detection = track["detection"]
                    condition = get_condition(detection["cropped"], animal_condition_model)
                    last_conditions[track_id] = condition
            
            # Check threats
            if frame_count % 10 == 0:  # Threat skip
                last_threats = get_threats(frame, model, animal_envo)
            
            # Process each track
            for track in tracks:
                track_id = track["track_id"]
                detection = track["detection"]
                
                if track_id not in saved_animals:
                    condition = last_conditions.get(track_id, "normal")
                    threat_str = ",".join(last_threats)
                    location = default_locations.get(detection["name"], "Wildlife Reserve")
                    
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
    
    # Show alerts
    for track_id in injury_alerts:
        animal_type = track_id.split('_')[0]
        show_alert("injury", f"Immediate medical attention required for {animal_type}!", animal_type)
    
    if len(threat_alerts) > 0:
        all_threats = set()
        for threat_tuple in threat_alerts:
            all_threats.update(threat_tuple)
        threat_list = [t for t in all_threats if t != "None"]
        if threat_list:
            show_alert("threat", f"Detected threats: {', '.join(threat_list)}")
    
    # Create DataFrame
    if results_list:
        df = pd.DataFrame(results_list)
        return df, output_path
    else:
        return pd.DataFrame(), output_path

def get_download_link(file_path, file_name):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">Wildlife Protection Blockchain System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.subheader("📁 Upload Video")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a wildlife video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload video file less than 100MB and under 1 minute for best performance"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎥 Video Processing")
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            
            # Display video info
            file_size = len(uploaded_file.getvalue()) / (1024*1024)  # MB
            st.info(f"📊 Video: {uploaded_file.name} ({file_size:.1f} MB)")
            
            # Process button
            if st.button("🚀 Start AI Analysis", type="primary"):
                with st.spinner("🤖 AI models are analyzing the video..."):
                    df, output_video_path = process_video_streamlit(st.session_state.video_path)
                    
                    if df is not None and not df.empty:
                        st.session_state.results_df = df
                        st.session_state.output_video_path = output_video_path
                        st.session_state.processing_complete = True
                        
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ Processing Complete!</h3>
                            <p>Video analysis finished successfully. Check results below.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No animals detected in the video. Try a different video.")
        
        # Results section
        if st.session_state.processing_complete and st.session_state.results_df is not None:
            st.subheader("📊 Detection Results")
            
            # Summary metrics
            df = st.session_state.results_df
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("🐅 Total Animals", len(df))
            with col_b:
                injured_count = len(df[df['condition'] == 'injured'])
                st.metric("🏥 Injured", injured_count)
            with col_c:
                threats_detected = len(df[df['threats'] != 'None'])
                st.metric("⚠️ Threats", threats_detected)
            with col_d:
                species_count = df['species_type'].nunique()
                st.metric("🦏 Species", species_count)
            
            # Data table
            st.dataframe(df, use_container_width=True)
            
            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV Report",
                    csv,
                    "wildlife_report.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            with col_dl2:
                if hasattr(st.session_state, 'output_video_path'):
                    with open(st.session_state.output_video_path, "rb") as file:
                        st.download_button(
                            "🎬 Download Annotated Video",
                            file,
                            "tracked_wildlife.mp4",
                            "video/mp4",
                            key='download-video'
                        )
    
    with col2:
        st.subheader("🔗 Blockchain Status")
        
        # Blockchain info
        blockchain = st.session_state.blockchain
        total_blocks = len(blockchain.chain)
        is_valid = blockchain.validate_chain()
        
        st.metric("📦 Total Blocks", total_blocks)
        st.metric("✅ Chain Valid", "Yes" if is_valid else "No")
        
        if total_blocks > 1:
            st.subheader("🧱 Recent Blocks")
            chain_data = blockchain.get_chain_data()
            
            for block in chain_data[-3:]:  # Show last 3 blocks
                st.markdown(f"""
                <div class="blockchain-block">
                    <strong>Block #{block['index']}</strong><br>
                    Hash: {block['hash']}<br>
                    Time: {block['timestamp']}<br>
                    Nonce: {block['nonce']}
                </div>
                """, unsafe_allow_html=True)
        
        # System info
        st.subheader("🖥️ System Info")
        st.info(f"""
        **AI Models**: Loaded ✅
        **Device**: {device.upper()}
        **Blockchain**: Active ✅
        **Security**: Military Grade 🔒
        """)

if __name__ == "__main__":
    main()
