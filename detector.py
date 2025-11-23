import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile

# -----------------------------------------
# Load models
# -----------------------------------------
# Ensure these model files are in the same directory or provide absolute paths
model_person = YOLO("my_model_cs.pt")      # soldier + civilian model
model_weapon = YOLO("my_model_w.pt")       # weapon model

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

            if cls1 == "civilian" and cls2 == "soldier":
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

# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------
st.set_page_config(page_title="Weapon Detection Dashboard", layout="wide")

st.title("🔫 Soldier–Civilian–Weapon Safety Detection")

mode = st.radio(
    "Select Mode",
    ["Upload Image", "Webcam Detection"],
    horizontal=True
)

SAFE_SOUND = "https://actions.google.com/sounds/v1/cartoon/pop.ogg"
ALERT_SOUND = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"

def play_sound(url):
    st.markdown(
        f"""
        <audio autoplay>
            <source src="{url}" type="audio/ogg">
        </audio>
        """,
        unsafe_allow_html=True
    )

def process_frame(img, model_person, model_weapon):
    """
    Processes a single frame (image) to detect objects and determine suspicion status.
    """
    img_detection = img.copy()
    
    # Run models
    results_person = model_person(img)[0]
    results_weapon = model_weapon(img)[0]

    detections = []
    soldiers, civilians, weapons = [], [], []

    # ----------------------------
    # 1. Process PERSON detections
    # ----------------------------
    for box in results_person.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results_person.names[cls_id].lower()

        det = {"cls": label, "conf": conf, "box": [x1, y1, x2, y2]}
        detections.append(det)

        if label == "soldier":
            soldiers.append(det)
        elif label == "civilian":
            civilians.append(det)

    # ----------------------------
    # 2. Process WEAPON detections
    # ----------------------------
    for box in results_weapon.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results_weapon.names[cls_id].lower()

        det = {"cls": label, "conf": conf, "box": [x1, y1, x2, y2]}
        detections.append(det)
        weapons.append(det)

    detections = suppress_overlaps(detections)

    # ----------------------------
    # 3. SUSPICION LOGIC (Refined)
    # ----------------------------
    suspicious = False
    
    for w in weapons:
        w_center = center(w["box"])
        
        # --- Step A: Spatial Containment ---
        # If weapon is physically inside a soldier's box but NOT a civilian's, assume Safe.
        in_soldier_box = any(point_in_box(w_center, s["box"]) for s in soldiers)
        in_civilian_box = any(point_in_box(w_center, c["box"]) for c in civilians)

        if in_soldier_box and not in_civilian_box:
            # Weapon is strictly held by soldier
            continue 

        # --- Step B: Distance & Dynamic Threshold ---
        # If containment is ambiguous (in both or neither), check distances.
        
        soldier_dist = float('inf')
        if soldiers:
            soldier_dist = min([distance(w_center, center(s["box"])) for s in soldiers])
            
        if civilians:
            # Find closest civilian to this weapon
            closest_c = min(civilians, key=lambda c: distance(w_center, center(c["box"])))
            c_dist = distance(w_center, center(closest_c["box"]))
            c_box = closest_c["box"]
            
            # Calculate Dynamic Threshold:
            # "Close" is defined relative to the person's size (height).
            # In a wide shot, people are small, so the threshold shrinks.
            # In a selfie, people are big, so the threshold grows.
            c_height = c_box[3] - c_box[1] 
            holding_radius = c_height * 0.4  # Threshold is 40% of body height
            
            # SUSPICION CRITERIA:
            # 1. Closer to civilian than soldier
            # 2. Actually close enough to be "held" (within holding_radius)
            if c_dist < soldier_dist and c_dist < holding_radius:
                suspicious = True
                cv2.line(img_detection, 
                         tuple(map(int, w_center)), 
                         tuple(map(int, center(c_box))), 
                         (0, 0, 255), 2)
                break
        
    status_text = "SUSPICIOUS" if suspicious else "SAFE"
    status_color = (0, 0, 255) if suspicious else (0, 255, 0)

    # Draw boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = det["cls"]
        conf = det["conf"]

        if label == "soldier":
            color = (0, 255, 0)
        elif label == "civilian":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(img_detection, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_detection, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_detection, status_text, status_color, suspicious


# =====================================================================
# 1️⃣  IMAGE UPLOAD MODE
# =====================================================================
if mode == "Upload Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    view_option = st.radio(
        "Select View:",
        ["Safety Result Only", "All Detections"],
        horizontal=True
    )

    if uploaded:
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, uploaded.name)

        with open(image_path, "wb") as f:
            f.write(uploaded.read())

        img = cv2.imread(image_path)
        
        # Process the image
        img_detection, status_text, status_color, is_suspicious = process_frame(img, model_person, model_weapon)

        # Create status image
        img_status = img.copy()
        cv2.putText(img_status, status_text, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 6)

        # Display Output
        if view_option == "Safety Result Only":
            st.image(img_status, channels="BGR")
            play_sound(ALERT_SOUND if is_suspicious else SAFE_SOUND)

        else:
            st.image(img_detection, channels="BGR")
            st.markdown(f"<h2 style='color: {'red' if is_suspicious else 'green'}'>Result: {status_text}</h2>", unsafe_allow_html=True)
            play_sound(ALERT_SOUND if is_suspicious else SAFE_SOUND)


# =====================================================================
# 2️⃣  WEBCAM MODE
# =====================================================================
if mode == "Webcam Detection":

    start_button = st.button("▶ Start Webcam")
    stop_button = st.button("⏹ Stop Webcam")

    FRAME_WINDOW = st.image([])
    
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
                st.error("Failed to capture video.")
                break

            img_detection, status_text, status_color, is_suspicious = process_frame(frame, model_person, model_weapon)
            
            cv2.putText(img_detection, status_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

            FRAME_WINDOW.image(img_detection, channels="BGR")

        cap.release()
