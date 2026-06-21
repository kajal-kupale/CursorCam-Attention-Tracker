# CursorCam-Attention-Tracker
# 🎯 CursorCam: ML-Based Attention and Engagement Tracker

## 📋 Project Description

CursorCam is A supervised Machine Learning based Attention and Engagement Tracking System developed as a Final Year Bachelor of Engineering (Computer Science & Engineering – Artificial Intelligence & Machine Learning) major project.

The system combines **Computer Vision**, **Behavioral Tracking**, **Feature Fusion**, and a **Random Forest Machine Learning Model** to analyze user attention in real-time. By continuously monitoring facial cues through a webcam and tracking mouse interaction patterns, CursorCam intelligently predicts whether a user is **ATTENTIVE** or **NOT ATTENTIVE** during digital learning activities.

The collected multimodal data is processed and fused into meaningful features, which are then classified using a trained Random Forest model. The results are presented through an interactive dashboard containing real-time attention status, assessment modules, learning resources, and analytical visualizations.

---

## 🚀 Core Working of CursorCam

### 🤖 Random Forest Classifier

A supervised Machine Learning model trained on multimodal attention data to classify user attention levels as **ATTENTIVE** or **NOT ATTENTIVE** with confidence scores.

### 🧠 Attention Prediction Engine

Analyzes real-time user behavior and continuously predicts engagement levels using extracted visual and behavioral features.

### 🔗 Feature Fusion

Combines facial features and mouse interaction data into a unified feature vector, improving prediction accuracy and robustness.

### 👁️ Computer Vision & Behavioral Tracking

Uses webcam-based facial analysis and mouse activity monitoring to detect:

- Face Presence Detection
- Eye Gaze Direction Tracking
- Eye State (Open / Closed)
- Head Pose Estimation
- Mouse Movement Analysis
- Mouse Click Activity
- User Interaction Patterns

---

## 🔍 Key Objectives

- Monitor user attention in real-time
- Detect engagement using facial and behavioral features
- Predict attention levels using Machine Learning
- Analyze learning behavior through multimodal data
- Generate visual analytics and attention reports
- Support intelligent digital learning environments

---


## ⚙️ Working Pipeline

```text
Webcam + Mouse Tracking
          ↓
   Feature Extraction
          ↓
Multimodal Feature Fusion
          ↓
Random Forest Classification
          ↓
   Attention Prediction
          ↓
Analytics Dashboard & Visualization
```


---

# 🌟 Key Highlights

## 🧠 Intelligent Attention Detection

### ✅ Real-Time Face Detection

- Continuous webcam monitoring
- Face presence verification

### ✅ Eye Gaze Tracking

- Center, left, right gaze detection
- Focus measurement

### ✅ Head Pose Analysis

- Head orientation tracking
- Engagement estimation

### ✅ Mouse Behavior Monitoring

- Cursor movement tracking
- Mouse click analysis
- Activity pattern detection


### ✅ Attention Trend Visualization

- Real-time attention analytics
- Interactive performance graphs
- Engagement monitoring

---


## 🔬 Machine Learning Integration

### ✅ Attention Classification Model

- Random Forest Classifier
- Feature Fusion Approach
- Confidence-Based Predictions

### ✅ Multimodal Analysis

- Facial Features
- Eye Gaze Data
- Head Pose Features
- Mouse Interaction Data
- Behavioral Activity Patterns

---
# # 🖥️ System Modules and Generated Outputs

### 📈 Analytics Dashboard

![Analytics Dashboard](Screenshots/Analytics%20Dashboard.png)

Visualizes attention scores, engagement trends, and learning behavior through interactive graphs.

---

### 🎥 Videos Section
![Learning Videos](Screenshots/Video%20Page.png)

Provides educational video resources that help users learn concepts while their attention level is monitored in real-time.

---

## 📚 Notes Section

![Study Notes](Screenshots/Notes%20Page.png)

Displays academic notes and learning materials for students.

---

### 📝 Assessment Module
![Assessment Module](Screenshots/Assessment%20Page.png)

Interactive quiz-based assessment system used to evaluate user understanding and engagement.

---

## 📊 Attention Dataset Samples

![Attention Dataset 1](Screenshots/Attention_Data_1.png)

![Attention Dataset 2](Screenshots/Attention_Data_2.png)

![Attention Dataset 3](Screenshots/Attention_Data_3.png)

 Displays sample records from the generated attention dataset containing facial features, eye gaze direction, eye state, head pose information, mouse activity, and attention labels. These records are used for training and evaluating the machine learning model for real-time attention classification.

---

### 🌟 Technology Integration

🧠 **Machine Learning-Powered Attention Prediction:**  
Employs a Random Forest Classifier to intelligently classify users as **ATTENTIVE** or **NOT ATTENTIVE** in real time.

👁️ **Advanced Computer Vision System:**  
Integrates OpenCV and MediaPipe to extract facial landmarks, eye gaze direction, eye state, and head pose features.

🔗 **Multimodal Behavioral Analytics:**  
Fuses visual attention indicators with mouse interaction patterns to improve prediction accuracy and engagement assessment.

📊 **Interactive Analytics & Learning Dashboard:**  
Delivers real-time attention status, confidence metrics, learning resources, assessments, and analytical insights through an intuitive Streamlit interface.

---


# 🛠️ Technology Stack

### 🤖 Machine Learning

![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-LATEST-orange)
![Random Forest](https://img.shields.io/badge/RANDOM_FOREST-CLASSIFIER-brightgreen)
![Attention Prediction](https://img.shields.io/badge/ATTENTION_PREDICTION-ML_MODEL-success)

### 👁️ Computer Vision

![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-blue)
![Face Tracking](https://img.shields.io/badge/FACE_TRACKING-ENABLED-0096D6)

### 📊 Data Processing

![Pandas](https://img.shields.io/badge/Pandas-2.3.0-purple)
![NumPy](https://img.shields.io/badge/NumPy-2.3.1-blue)

### 🖱️ Tracking & Analytics

![Pynput](https://img.shields.io/badge/PYNPUT-MOUSE_TRACKING-red)
![Plotly](https://img.shields.io/badge/PLOTLY-6.1.2-darkblue)
![Interactive Charts](https://img.shields.io/badge/INTERACTIVE_CHARTS-ENABLED-blue)

### 🌐 Dashboard & UI

![Streamlit](https://img.shields.io/badge/STREAMLIT-1.46.0-FF4B4B)
![HTML5](https://img.shields.io/badge/HTML5-LATEST-E34F26)
![CSS3](https://img.shields.io/badge/CSS3-LATEST-1572B6)

### 🛠️ Development Tools

![Python](https://img.shields.io/badge/PYTHON-3.13-yellow)
![VS Code](https://img.shields.io/badge/VS_CODE-EDITOR-007ACC)
![Git](https://img.shields.io/badge/GIT-VERSION_CONTROL-F05032)
![GitHub](https://img.shields.io/badge/GITHUB-REPOSITORY-black)


---



## 📥 Installation Guide

### Requirements

- Python 3.13+
- Webcam (Required)
- 8 GB RAM (16 GB Recommended)

### Setup Instructions

```bash
# Clone Repository
git clone https://github.com/kajal-kupale/CursorCam-Attention-Tracker.git

# Open Project Directory
cd "CursorCam Project"

# Create Virtual Environment
python -m venv venv

# Activate Virtual Environment
venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Run Application
streamlit run app.py
```








