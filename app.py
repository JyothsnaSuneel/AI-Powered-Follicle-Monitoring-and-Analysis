import streamlit as st
from ultralytics import YOLO
import cv2, os, tempfile, torch
import numpy as np, pandas as pd
from datetime import datetime

# UI THEME + HEADER 
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

import gdown
import os

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_best_yolov8m_tuned.pt"
)

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    url = "https://drive.google.com/uc?id=1GJdIBwi7ieubyA4pYCS2JxXhk1dVPKeJ"

    with st.spinner("Loading AI model..."):
    gdown.download(...)
    gdown.download(url, MODEL_PATH, quiet=False)
    
CALIB_PATH = os.path.join(BASE_DIR, "calibration.txt")

OUT_DIR = "/tmp/outputs"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    
#  Page Setup 
st.set_page_config(page_title="AI-Powered Follicle Monitoring & Analysis", page_icon="🩺", layout="wide")

st.markdown("""
<style>
/* General layout */
[data-testid="stAppViewContainer"] {
    background-color: #f8fbff;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #e3f2fd;
    border-right: 2px solid #bbdefb;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
    color: #0d47a1 !important;
}

/* Header bar */
.title-bar {
    background: linear-gradient(90deg, #1976d2 0%, #2196f3 100%);
    padding: 18px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* Upload box */
.uploadbox {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    background-color: #1565c0 !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #0d47a1 !important;
}

/* Results video */
video {
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* Sidebar cards */
.info-card {
    background-color: #bbdefb;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: #0d47a1;
    font-weight: 500;
    text-align: center;
}

/* Footer */
.footer {
    position: relative;
    left: 0;
    bottom: 0;
    width: 100%;
    background: #1976d2;
    color: white;
    text-align: center;
    padding: 8px;
    font-size: 13px;
    border-top: 1px solid #1565c0;
}
</style>
""", unsafe_allow_html=True)

# HEADER 
st.markdown("<div class='title-bar'>🧠 AI-Powered Follicle Monitoring & Analysis</div>", unsafe_allow_html=True)
st.markdown("""
This AI-powered application automatically detects ovarian follicles from ultrasound videos,
estimates follicle diameter, and classifies maturity status to assist IVF monitoring.
""")
st.info("""
🎯 Model Performance:
• YOLOv8m Segmentation
• mAP@50: 95%
• Real-time follicle detection
• Automated diameter measurement
• Maturity classification support
""")

# Calibration
try:
    with open(CALIB_PATH, "r") as f:
        PIXELS_PER_MM = float(f.read().strip())
except:
    PIXELS_PER_MM = 6.10

#  Load YOLO Model 
@st.cache_resource
def load_model():
    torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
    model = YOLO(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model

model = load_model()
st.sidebar.success("✅ Model Loaded Successfully")

#  Helper 
def classify_maturity(d_mm):
    if d_mm < 18:
        return "Immature"
    elif 18 <= d_mm <= 20:
        return "Mature"
    else:
        return "Overmature"

#  Analyze Video 
def analyze_video(video_path, model, px_per_mm):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Faster: process every nth frame
    frame_skip = max(1, fps // 5)  # about 5 fps

    out_path = os.path.join(OUT_DIR, f"output_{datetime.now().strftime('%H%M%S')}.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    results_list = []
    progress = st.progress(0)
    frame_no = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_no % frame_skip != 0:
            continue

        # Predict frame
        results = model.predict(frame, conf=0.6, imgsz=640, verbose=False, device='cpu')

        for r in results:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):
                mask_uint8 = (mask > 0).astype(np.uint8)
                mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)

                area_px = np.sum(mask_resized)
                if area_px == 0:
                    continue

                diam_mm = round(np.sqrt((4 * area_px) / np.pi) / px_per_mm, 2)
                maturity = classify_maturity(diam_mm)

                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

                M = cv2.moments(mask_resized)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv2.putText(frame, f"{diam_mm} mm", (cx - 30, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                results_list.append([frame_no, i+1, diam_mm, maturity])

        out.write(frame)
        processed_frames += 1
        progress.progress(min(processed_frames * frame_skip / total_frames, 1.0))

    cap.release()
    out.release()

    df = pd.DataFrame(results_list, columns=["Frame", "Follicle_ID", "Diameter_mm", "Maturity"])
    csv_path = os.path.join(OUT_DIR, f"results_{datetime.now().strftime('%H%M%S')}.csv")
    df.to_csv(csv_path, index=False)

    return out_path, csv_path, df

#  Streamlit UI 
# SIDEBAR (Clinical Info) 

if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=140)

st.sidebar.markdown("### 🩺 Clinical Information")
st.sidebar.markdown("<div class='info-card'>Enter patient details below for record keeping</div>", unsafe_allow_html=True)

patient_name = st.sidebar.text_input("👩 Patient Name")
patient_id = st.sidebar.text_input("🧾 Patient ID")
clinician = st.sidebar.text_input("👨‍⚕️ Clinician")
exam_date = st.sidebar.date_input("📅 Exam Date", datetime.now().date())
pix_per_mm = st.sidebar.number_input("📏 Calibration (px/mm)", value=6.10, step=0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.caption("Developed for IVF Follicle Monitoring using AI-based segmentation.")

uploaded_video = st.file_uploader("🎥 Upload Ultrasound Video", type=["mp4", "avi", "mov"])
analyze = st.button("🔍 Analyze Video")

if uploaded_video and analyze:
    st.info("⏳ Processing video — please wait 2–3 minutes...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video.read())
        temp_video_path = tfile.name

    with st.spinner("Analyzing follicles..."):
        output_path, csv_path, df = analyze_video(temp_video_path, model, pix_per_mm)

    st.success("✅ Analysis Complete!")

    # Display processed video
    with open(output_path, "rb") as f:
        st.video(f.read())

    # Download buttons
    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Annotated Ultrasound Video", f.read(), file_name=os.path.basename(output_path), mime="video/mp4")

    st.subheader("📊 Results CSV")
    st.dataframe(df)
    st.download_button("⬇ Download CSV Results", data=df.to_csv(index=False), file_name=os.path.basename(csv_path))

else:
    st.info("📤 Upload your ultrasound video and click **Analyze Video** to start.")

#  FOOTER 
st.markdown("""
<div class='footer'>
🧠 Developed by <b>Jyothsna Suneel</b> | Powered by <b>AI for Healthcare</b> | 2025
</div>
""", unsafe_allow_html=True)











