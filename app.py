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
from streamlit_webrtc import webrtc_streamer, WebRTCStreamerContext

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

# Input type selection
input_type = st.sidebar.radio(
    "Select Input Source:",
    ("Image Upload", "Video Upload", "Webcam")
)

# Throttling variables
if 'last_publish_time' not in st.session_state:
    st.session_state.last_publish_time = 0
PUBLISH_INTERVAL = 3  # Seconds

def process_frame(frame: np.ndarray):
    """Process frame and return annotated image"""
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
    
    return results[0].plot()

# ===== Image Upload =====
if input_type == "Image Upload":
    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if img_file is not None:
        image = Image.open(img_file)
        image_np = np.array(image)
        processed_image = process_frame(image_np)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

# ===== Video Upload =====
elif input_type == "Video Upload":
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame(frame)
            stframe.image(processed_frame, channels="BGR")
        
        cap.release()

# ===== Webcam =====
elif input_type == "Webcam":
    st.warning("⚠️ Allow browser camera access when prompted")
    
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRTCStreamerContext.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_frame_callback=lambda frame: process_frame(frame.to_ndarray(format="bgr24")),
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 1280},
                "height": {"min": 480, "ideal": 720}
            },
            "audio": False
        },
        async_processing=True
    )

    if ctx.state.playing:
        st.info("Live webcam processing active! Detection results appear below.")

# Cleanup when app stops
def on_app_close():
    if 'mqttc' in st.session_state:
        st.session_state.mqttc.loop_stop()
        st.session_state.mqttc.disconnect()

import atexit
atexit.register(on_app_close)