import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
import csv
from datetime import datetime
from pathlib import Path

cv2.setNumThreads(0)

st.set_page_config(page_title="Face Emotion Recognition", layout="centered")
st.title("Face Emotion Recognition App")

st.write("Upload an image or use webcam. App will detect faces and predict emotions (uses DeepFace).")

# ---------- Input mode: Upload or Webcam ----------
mode = st.radio("Choose input", ("Upload Image", "Use Webcam"))

img = None
if mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload Image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
elif mode == "Use Webcam":
    cam_file = st.camera_input("Take a photo with camera", key="cam")
    if cam_file is not None:
        file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

if img is None:
    st.info("Upload image or take a photo to start.")
    st.stop()

# Show original
st.subheader("Original Image")
st.image(img, channels="BGR", use_column_width=True)

# Face detection (OpenCV Haar cascade)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    st.warning("No face Detected. Try again with clear Shot. ")
    st.stop()

st.success(f"Detected {len(faces)} face(s). Predicting emotions...")

results_list = []
for i, (x, y, w, h) in enumerate(faces):
    pad = int(0.1 * w)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)

    face_img_bgr = img[y1:y2, x1:x2]
    face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)

    try:
        # use opencv backend to avoid TensorFlow RetinaFace issues
        analysis = DeepFace.analyze(
            face_img_rgb,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if isinstance(analysis, list):
            analysis = analysis[0]
        emotion = analysis.get('dominant_emotion', None)
        emotion_scores = analysis.get('emotion', {})
    except Exception as e:
        emotion = None
        emotion_scores = {}
        st.error(f"DeepFace error on face {i+1}: {str(e)}")

    label = emotion if emotion is not None else "Unknown"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    results_list.append({
        "face_index": i+1,
        "emotion": emotion,
        "scores": emotion_scores
    })

# Show annotated image
st.subheader("Detected faces and labels")
st.image(img, channels="BGR", use_column_width=True)

# Show per-face emotion scores & bar chart
for res in results_list:
    st.markdown(f"### Face #{res['face_index']}: **{res['emotion']}**")
    if res['scores']:
        df = pd.DataFrame(list(res['scores'].items()), columns=["emotion", "score"])
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        st.table(df)            # table view
        st.bar_chart(df.set_index("emotion"))  # bar chart

# Save results to CSV
out_file = Path(__file__).parent / "results.csv"
header = ["timestamp", "face_index", "dominant_emotion", "scores_json"]
write_header = not out_file.exists()

with open(out_file, "a", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    for res in results_list:
        row = [
            datetime.now().isoformat(),
            res["face_index"],
            res["emotion"],
            str(res["scores"])
        ]
        writer.writerow(row)

st.success(f"Saved {len(results_list)} result(s) to results.csv (in project folder)")
st.write("Results are stored in `results.csv` inside the project folder. Each run appends new entries to the file.")