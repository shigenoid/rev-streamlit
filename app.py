import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import paho.mqtt.client as mqtt
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize MQTT client
if 'mqttc' not in st.session_state:
    try:
        st.session_state.mqttc = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=os.getenv("CLIENT_ID")
        )
        st.session_state.mqttc.connect(
            os.getenv("MQTT_SERVER"),
            int(os.getenv("MQTT_PORT"))
        )
        st.session_state.mqttc.loop_start()
    except Exception as e:
        st.error(f"MQTT Connection Error: {str(e)}")

model = YOLO('models/model-1.pt')

st.title("🍾 Revoira Object Detection")

# ===== MQTT Status Section =====
with st.sidebar.expander("🔌 MQTT Connection Status", expanded=True):
    try:
        status = "✅ Connected" if st.session_state.mqttc.is_connected() else "❌ Disconnected"
        st.metric(label="MQTT Status", value=status)
        st.caption(f"Broker: {os.getenv('MQTT_SERVER')}:{os.getenv('MQTT_PORT')}")
        st.caption(f"Topic: {os.getenv('MQTT_TOPIC')}")
        st.caption(f"Client ID: {os.getenv('CLIENT_ID')}")
    except Exception as e:
        st.error(f"MQTT Status Error: {str(e)}")

# ===== Model Information Section =====
with st.expander("⚠️ Important Model Notes", expanded=True):
    st.markdown("""
    **Model Specifications:**
                
    This model's purpose was to detect a type of bottles based on its material 
                
    - **List of classes:** 
      - 🥫 can-bottle | 🧴 plastic-bottle | 🍾 glass-bottle | 📦 tetrapak
                
    **Current Limitations:**
                
    Keep in mind that the models will be used in an environment where only such conditions will exists so there will
    be some limitations such as:
                
    1. **Single-Class Detection Preference**  
       The model tends to detect only the most prominent object in a frame when multiple objects are present.  
    
    2. **Optimal Detection Conditions**  
       Works best with:
       - Single objects centered in frame
       - Flat, uniform backgrounds (like conveyor belts)
       - Good lighting conditions
       - Objects placed on solid surfaces
    
    3. **Performance Notes**  
       Detection quality could decreases when:
       - Objects overlap or are too close
       - Backgrounds are cluttered
       - Lighting is uneven or creates shadows
    """)

st.warning("Also you can change the model to model-2.pt if model-1.pt seemed to be more unstable (change it in app.py)")

# Input type selection
input_type = st.sidebar.radio(
    "Select Input Source:",
    ("Image Upload", "Video Upload", "Webcam")
)

# Throttling variables
if 'last_publish_time' not in st.session_state:
    st.session_state.last_publish_time = 0
PUBLISH_INTERVAL = 3  # Seconds

def process_frame(frame):
    """Process frame and return both results and annotated image"""
    results = model.predict(frame, conf=0.6)
    detected_classes = set()
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            detected_classes.add(model.names[cls_id])
    
    # Throttled MQTT Publishing
    current_time = time.time()
    if current_time - st.session_state.last_publish_time >= PUBLISH_INTERVAL:
        msg = ",".join(detected_classes) if detected_classes else "NONE"
        try:
            st.session_state.mqttc.publish(os.getenv("MQTT_TOPIC"), msg)
            st.session_state.last_publish_time = current_time
            st.sidebar.success(f"Published: {msg}")
        except Exception as e:
            st.sidebar.error(f"MQTT Publish Error: {str(e)}")
    
    return results[0].plot(), detected_classes

# ===== Image Upload =====
if input_type == "Image Upload":
    img_file = st.file_uploader(
        "Upload Image", 
        type=["jpg", "jpeg", "png"]
    )
    
    if img_file is not None:
        image = Image.open(img_file)
        image_np = np.array(image)
        
        processed_image, detected = process_frame(image_np)
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        st.write(f"**Detected Objects:** {', '.join(detected) if detected else 'No objects detected'}")

# ===== Video Upload =====
elif input_type == "Video Upload":
    video_file = st.file_uploader(
        "Upload Video", 
        type=["mp4", "avi", "mov"]
    )
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, detected = process_frame(frame)
            stframe.image(processed_frame, channels="BGR")
        
        cap.release()

# ===== Webcam =====
elif input_type == "Webcam":
    st.warning("⚠️ Webcam access requires browser permission")
    
    if 'cam_active' not in st.session_state:
        st.session_state.cam_active = False
    
    video_placeholder = st.empty()
    
    start_button, stop_button = st.columns(2)
    
    with start_button:
        if st.button("Start Webcam"):
            st.session_state.cam_active = True
    
    with stop_button:
        if st.button("Stop Webcam"):
            st.session_state.cam_active = False
    
    if st.session_state.cam_active:
        cap = cv2.VideoCapture(0)
        
        try:
            while st.session_state.cam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                frame = cv2.flip(frame, 1)
                processed_frame, detected = process_frame(frame)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(processed_frame_rgb, channels="RGB")
                
        finally:
            cap.release()
            st.session_state.cam_active = False

# Cleanup when app stops
def on_app_close():
    if 'mqttc' in st.session_state:
        st.session_state.mqttc.loop_stop()
        st.session_state.mqttc.disconnect()

import atexit
atexit.register(on_app_close)