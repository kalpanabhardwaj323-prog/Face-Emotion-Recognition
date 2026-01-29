# Face Emotion Recognition System

A complete Face Emotion Recognition application built using Python, Streamlit, OpenCV and DeepFace.
This system detects human faces from images or webcam and predicts the dominant emotion in real-time.

---

## 📌 Project Overview
This project uses computer vision and deep learning to analyze facial expressions and classify emotions such as:
Happy, Sad, Angry, Surprise, Fear, Disgust and Neutral.

It is useful in:
• Human–Computer Interaction  
• Customer behaviour analysis  
• Mental health monitoring  
• Smart surveillance systems  

---

## 🔧 Tech Stack
• Python  
• Streamlit  
• OpenCV  
• DeepFace  
• NumPy  
• Pandas  
• TensorFlow  

---

## ✨ Features
• Upload image or use webcam  
• Automatic face detection  
• Emotion classification  
• Emotion confidence scores  
• Visual bounding boxes  
• Bar chart visualization  
• Result logging in CSV  

---

## ⚙ How it Works
1. User uploads an image or captures from webcam  
2. OpenCV detects face regions  
3. DeepFace analyzes emotions  
4. Dominant emotion is displayed  
5. Results are saved in CSV file  

---

## ▶ How to Run

### Step 1: Install dependencies
pip install -r requirements.txt

### Step 2: Run application
streamlit run app.py

---

## 📂 Output
All detected emotions are stored in:
results.csv
Each run appends new records with timestamp, face index and emotion scores.

---

## 👤 Author
Kalpana Bhardwaj  
BTech CSE (AI & ML)  
LinkedIn: https://www.linkedin.com/in/kalpana-bhardwaj-1551b52b2/
