import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
from datetime import datetime

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="AI Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------
# Custom CSS Styling
# -----------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2a1a3f 100%);
        background-attachment: fixed;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header Section */
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.5);
    }
    
    .subtitle {
        color: #a8b2d1;
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Control Panel */
    .control-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .stat-card:hover::before {
        left: 100%;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .stat-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .stat-number {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
    }
    
    .stat-label {
        color: #a8b2d1;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Detection Result */
    .result-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0;
        animation: statusPulse 2s ease-in-out infinite;
    }
    
    .status-safe {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.5);
    }
    
    .status-alert {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
        animation: alertPulse 1s ease-in-out infinite;
    }
    
    @keyframes statusPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes alertPulse {
        0%, 100% { 
            transform: scale(1); 
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
        }
        50% { 
            transform: scale(1.08); 
            box-shadow: 0 0 50px rgba(239, 68, 68, 0.8);
        }
    }
    
    /* Legend */
    .legend-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        backdrop-filter: blur(10px);
        border: 2px solid;
        transition: all 0.3s ease;
    }
    
    .legend-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    .legend-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
    }
    
    .legend-soldier {
        border-color: #10b981;
    }
    
    .legend-soldier .legend-dot {
        background: #10b981;
        box-shadow: 0 0 15px #10b981;
    }
    
    .legend-civilian {
        border-color: #fbbf24;
    }
    
    .legend-civilian .legend-dot {
        background: #fbbf24;
        box-shadow: 0 0 15px #fbbf24;
    }
    
    .legend-weapon {
        border-color: #ef4444;
    }
    
    .legend-weapon .legend-dot {
        background: #ef4444;
        box-shadow: 0 0 15px #ef4444;
    }
    
    .legend-text {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2.5rem;
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    /* Info Box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #a8b2d1;
        line-height: 1.6;
    }
    
    .info-box strong {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Load Models
# -----------------------------------------
@st.cache_resource
def load_models():
    model_person = YOLO("my_model_cs.pt")
    model_weapon = YOLO("my_model_w.pt")
    return model_person, model_weapon

try:
    model_person, model_weapon = load_models()
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")
    st.stop()

# -----------------------------------------
# Utility Functions
# -----------------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def suppress_overlaps(detections):
    final = []
    for det in detections:
        cls1, box1, conf1 = det["cls"], det["box"], det["conf"]
        remove = False
        for other in detections:
            if det == other:
                continue
            cls2, box2, conf2 = other["cls"], other["box"], other["conf"]
            # Check for both "civilian" and "soldier" variations
            if cls1 in ["civilian", "civillian"] and cls2 in ["soldier", "solider"]:
                if compute_iou(box1, box2) > 0.20 and conf2 > conf1:
                    remove = True
                    break
        if not remove:
            final.append(det)
    return final

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(c1, c2):
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5

def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def process_frame(img, model_person, model_weapon):
    img_detection = img.copy()
    results_person = model_person(img)[0]
    results_weapon = model_weapon(img)[0]

    detections = []
    soldiers, civilians, weapons = [], [], []

    for box in results_person.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results_person.names[cls_id].lower()
        det = {"cls": label, "conf": conf, "box": [x1, y1, x2, y2]}
        detections.append(det)
        # Handle both "soldier" and "solider" (common typo in datasets)
        if label in ["soldier", "solider"]:
            soldiers.append(det)
        elif label in ["civilian", "civillian"]:
            civilians.append(det)

    for box in results_weapon.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results_weapon.names[cls_id].lower()
        det = {"cls": label, "conf": conf, "box": [x1, y1, x2, y2]}
        detections.append(det)
        weapons.append(det)

    detections = suppress_overlaps(detections)
    
    # Recount after suppression - handle variations in class names
    soldiers = [d for d in detections if d["cls"] in ["soldier", "solider"]]
    civilians = [d for d in detections if d["cls"] in ["civilian", "civillian"]]
    weapons = [d for d in detections if d["cls"] not in ["soldier", "solider", "civilian", "civillian"]]

    suspicious = False
    
    for w in weapons:
        w_center = center(w["box"])
        in_soldier_box = any(point_in_box(w_center, s["box"]) for s in soldiers)
        in_civilian_box = any(point_in_box(w_center, c["box"]) for c in civilians)

        if in_soldier_box and not in_civilian_box:
            continue 

        soldier_dist = float('inf')
        if soldiers:
            soldier_dist = min([distance(w_center, center(s["box"])) for s in soldiers])
            
        if civilians:
            closest_c = min(civilians, key=lambda c: distance(w_center, center(c["box"])))
            c_dist = distance(w_center, center(closest_c["box"]))
            c_box = closest_c["box"]
            c_height = c_box[3] - c_box[1] 
            holding_radius = c_height * 0.4
            
            if c_dist < soldier_dist and c_dist < holding_radius:
                suspicious = True
                cv2.line(img_detection, 
                         tuple(map(int, w_center)), 
                         tuple(map(int, center(c_box))), 
                         (0, 0, 255), 3)
                break

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = det["cls"]
        conf = det["conf"]

        # Normalize label for display
        if label in ["soldier", "solider"]:
            color = (0, 255, 0)
            display_label = "SOLDIER"
        elif label in ["civilian", "civillian"]:
            color = (0, 255, 255)
            display_label = "CIVILIAN"
        else:
            color = (0, 0, 255)
            display_label = label.upper()

        cv2.rectangle(img_detection, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_detection, f"{display_label} {conf:.0%}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img_detection, suspicious, len(soldiers), len(civilians), len(weapons)

SAFE_SOUND = "https://actions.google.com/sounds/v1/cartoon/pop.ogg"
ALERT_SOUND = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"

def play_sound(url):
    st.markdown(f'<audio autoplay><source src="{url}" type="audio/ogg"></audio>', unsafe_allow_html=True)

# -----------------------------------------
# HEADER
# -----------------------------------------
st.markdown("""
<div class="header-container">
    <h1 class="main-title">🛡️ AI Security Detection System</h1>
    <p class="subtitle">Advanced Threat Assessment & Real-time Monitoring</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# LEGEND
# -----------------------------------------
st.markdown("""
<div class="legend-container">
    <div class="legend-item legend-soldier">
        <div class="legend-dot"></div>
        <span class="legend-text">Soldier</span>
    </div>
    <div class="legend-item legend-civilian">
        <div class="legend-dot"></div>
        <span class="legend-text">Civilian</span>
    </div>
    <div class="legend-item legend-weapon">
        <div class="legend-dot"></div>
        <span class="legend-text">Weapon</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# MODE SELECTION
# -----------------------------------------
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode = st.radio(
        "Select Detection Mode",
        ["📸 Upload Image", "🎥 Live Webcam"],
        horizontal=True,
        label_visibility="collapsed"
    )
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# IMAGE UPLOAD MODE
# =====================================================================
if mode == "📸 Upload Image":
    
    uploaded = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Display Options")
            view_option = st.radio(
                "Select view",
                ["🎯 Full Detection", "📋 Status Only"],
                horizontal=False
            )
            
            st.markdown("""
            <div class="info-box">
                <strong>🔍 Detection Logic:</strong><br>
                The system flags suspicious activity when weapons are detected in close proximity to civilians rather than authorized military personnel.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, uploaded.name)
            with open(image_path, "wb") as f:
                f.write(uploaded.read())
            
            img = cv2.imread(image_path)
            
            with st.spinner("🔍 Analyzing image..."):
                img_detection, is_suspicious, n_soldiers, n_civilians, n_weapons = process_frame(
                    img, model_person, model_weapon
                )
            
            # Display image
            if view_option == "📋 Status Only":
                img_status = img.copy()
                status_text = "SUSPICIOUS ACTIVITY" if is_suspicious else "AREA SECURE"
                status_color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                cv2.putText(img_status, status_text, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 6)
                st.image(img_status, channels="BGR", use_container_width=True)
            else:
                st.image(img_detection, channels="BGR", use_container_width=True)
            
            # Status Badge
            if is_suspicious:
                st.markdown('<div style="text-align: center;"><span class="status-badge status-alert">⚠️ SUSPICIOUS ACTIVITY DETECTED</span></div>', unsafe_allow_html=True)
                play_sound(ALERT_SOUND)
            else:
                st.markdown('<div style="text-align: center;"><span class="status-badge status-safe">✓ AREA SECURE</span></div>', unsafe_allow_html=True)
                play_sound(SAFE_SOUND)
        
        # Stats Section - Full Width
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">🎖️</div>
                <div class="stat-number">{n_soldiers}</div>
                <div class="stat-label">Soldiers Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">👥</div>
                <div class="stat-number">{n_civilians}</div>
                <div class="stat-label">Civilians Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🔫</div>
                <div class="stat-number">{n_weapons}</div>
                <div class="stat-label">Weapons Detected</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        st.markdown(f'<p style="text-align: center; color: #a8b2d1; margin-top: 1rem;">⏰ Analysis completed at {timestamp}</p>', unsafe_allow_html=True)

# =====================================================================
# WEBCAM MODE
# =====================================================================
elif mode == "🎥 Live Webcam":
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_button = st.button("▶️ Start Detection", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
    
    # Stats placeholders
    stats_placeholder = st.empty()
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False
    
    if start_button:
        st.session_state.run_webcam = True
    if stop_button:
        st.session_state.run_webcam = False
    
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        
        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture video.")
                break
            
            img_detection, is_suspicious, n_soldiers, n_civilians, n_weapons = process_frame(
                frame, model_person, model_weapon
            )
            
            status_text = "SUSPICIOUS" if is_suspicious else "SECURE"
            status_color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            cv2.putText(img_detection, status_text, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
            
            # Display stats
            stats_placeholder.markdown(f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">🎖️</div>
                    <div class="stat-number">{n_soldiers}</div>
                    <div class="stat-label">Soldiers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">👥</div>
                    <div class="stat-number">{n_civilians}</div>
                    <div class="stat-label">Civilians</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🔫</div>
                    <div class="stat-number">{n_weapons}</div>
                    <div class="stat-label">Weapons</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display video
            video_placeholder.image(img_detection, channels="BGR", use_container_width=True)
            
            # Display status
            if is_suspicious:
                status_placeholder.markdown('<div style="text-align: center;"><span class="status-badge status-alert">⚠️ SUSPICIOUS ACTIVITY</span></div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown('<div style="text-align: center;"><span class="status-badge status-safe">✓ AREA SECURE</span></div>', unsafe_allow_html=True)
        
        cap.release()
