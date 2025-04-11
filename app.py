import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

model = YOLO('models/model-1.pt')

st.title("🍾 Revoira Object Detection")

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
       Detection quality decreases when:
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

def process_frame(frame):
    """Process a single frame with YOLOv8"""
    results = model.predict(frame)  # Confidence threshold removed
    return results[0].plot()

# ===== Image Upload =====
if input_type == "Image Upload":
    img_file = st.file_uploader(
        "Upload Image", 
        type=["jpg", "jpeg", "png"]
    )
    
    if img_file is not None:
        image = Image.open(img_file)
        image_np = np.array(image)
        
        # Process and display
        processed_image = process_frame(image_np)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

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
            
            # Process frame
            processed_frame = process_frame(frame)
            stframe.image(processed_frame, channels="BGR")
        
        cap.release()

# ===== Webcam =====
elif input_type == "Webcam":
    st.warning("⚠️ Webcam access requires browser permission")
    
    # Initialize session state
    if 'cam_active' not in st.session_state:
        st.session_state.cam_active = False
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    start_button, stop_button = st.columns(2)
    
    with start_button:
        if st.button("Start Webcam"):
            st.session_state.cam_active = True
    
    with stop_button:
        if st.button("Stop Webcam"):
            st.session_state.cam_active = False
    
    # Webcam processing loop
    if st.session_state.cam_active:
        cap = cv2.VideoCapture(0)
        
        try:
            while st.session_state.cam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = process_frame(frame)
                
                # Convert to RGB for Streamlit
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update the single placeholder
                video_placeholder.image(processed_frame_rgb, channels="RGB")
                
        finally:
            cap.release()
            st.session_state.cam_active = False