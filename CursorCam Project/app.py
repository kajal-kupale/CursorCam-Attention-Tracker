# File: MP_Proj/app.py
import streamlit as st
from pathlib import Path
import time
from collections import deque
import pandas as pd


# Initialize session state for lag feature
if "attention_continuity" not in st.session_state:
    st.session_state.attention_continuity = 0

from utils.webcam_capture import start_capture, stop_capture
from utils.realtime_prediction import predict_attention


# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🧠 User Attention Tracker",
    page_icon="🧠",
    layout="wide",
)


# ====================== CUSTOM CSS ======================

st.markdown("""
<style>

/* ==========================================
   MAIN APPLICATION
========================================== */

.stApp{
    background-color:#F5F7FA;
}

/* ==========================================
   PROFESSIONAL HEADER
========================================== */

header[data-testid="stHeader"]{
    background-color:#273748 !important;
}

/* Hide Streamlit Toolbar */
[data-testid="stToolbar"]{
    visibility:hidden;
}

/* ==========================================
   MAIN CONTENT
========================================== */

.block-container{
    max-width:1000px;
    margin:auto;
    padding-top:2rem;
    padding-bottom:0.5rem;
}

/* ==========================================
   SIDEBAR
========================================== */

section[data-testid="stSidebar"]{
    background-color:#001D39;
    width:260px !important;
}

/* ==========================================
   SIDEBAR TEXT
========================================== */

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* ==========================================
   SELECT SECTION TITLE
========================================== */

/* ==========================================
   NAVIGATION TITLE
========================================== */

label[data-testid="stWidgetLabel"] p{
    color:white !important;
    font-size:24px !important;
    font-weight:500 !important;
}
/* ==========================================
   NAVIGATION BOX
========================================== */

.stRadio div[role="radiogroup"]{
    background-color:white;
    border-radius:10px;
    border:1px solid #D1D5DB;

    padding:12px;
    width:170px;       /* wider box */
}

/* ==========================================
   NAVIGATION OPTIONS
========================================== */

div[role="radiogroup"] label p{
    color:black !important;
    font-size:15px;
    font-weight:600;
}

/* ==========================================
   HEADINGS
========================================== */

h1{
    color:black !important;
    font-size:2rem !important;
    font-weight:700 !important;
}

h2{
    color:#1E293B !important;
    font-size:1.35rem !important;
    font-weight:600 !important;
}

h3,h4,h5,h6{
    color:#1E293B !important;
    font-weight:600 !important;
}

/* ==========================================
   NORMAL TEXT
========================================== */

p,label{
    color:#111827 !important;
}

/* ==========================================
   STOP TRACKER BUTTON
========================================== */

.stButton button{
    background-color:#EF233C !important;
    color:white !important;
    border:none !important;
    border-radius:8px !important;
    width:100%;
    height:45px;
    font-size:15px;
    font-weight:600;
}

/* ==========================================
   BUTTON HOVER
========================================== */

.stButton button:hover{
    background-color:#B91C1C !important;
}

/* ==========================================
   RADIO OPTIONS
========================================== */

.stRadio label{
    color:black !important;
    font-size:14px;
    font-weight:500;
}

/* ==========================================
   SUCCESS / WARNING / ERROR BOXES
========================================== */

div[data-testid="stAlert"]{
    border-radius:10px;
    font-weight:600;
}

/* ==========================================
   VIDEO
========================================== */

video{
    border-radius:8px;
}

/* ==========================================
   REDUCE GAP BETWEEN QUESTIONS
========================================== */

.element-container{
    margin-bottom:-0.1rem !important;
}

/* ==========================================
   PLOTLY GRAPH
========================================== */

.js-plotly-plot{
    margin:auto;
}

/* ==========================================
   SCROLLBAR
========================================== */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#94A3B8;
    border-radius:10px;
}

/* ==========================================
   ANALYTICS CAPTION
========================================== */

small{
    color:#E91E63 !important;
    font-weight:600 !important;
}


/* Sidebar compact */
section[data-testid="stSidebar"]{
    width:240px !important;
}

/* Remove sidebar scrolling */
section[data-testid="stSidebar"] > div:first-child{
    overflow-y:hidden !important;
}

/* Navigation options */
div[role="radiogroup"]{
    padding:8px !important;
}

/* Radio spacing */
div[role="radiogroup"] label{
    margin-bottom:4px !important;
}

/* Assessment radio buttons */
.stRadio > div{
    gap:2px !important;
}

/* Text area */
textarea{
    border-radius:8px !important;
}

/* Success box */
div[data-testid="stAlert"]{
    border-radius:10px !important;
}

.stRadio div[role="radiogroup"]{
    background-color:white;
    border-radius:10px;
    border:1px solid #D1D5DB;
    padding:8px;

    margin-top:0px !important;
}





</style>
""", unsafe_allow_html=True)


# ====================== BACKGROUND TRACKER ======================
if "tracker_running" not in st.session_state:
    start_capture()
    st.session_state.tracker_running = True

st.sidebar.markdown("""
<div style="
background-color:#235347;
padding:10px;
border-radius:4px;
font-weight:600;
color:#15803D;">
✅ Attention Tracker running in background
</div>
""", unsafe_allow_html=True)

# GAP
st.sidebar.markdown(
"""
<div style='height:-8px'></div>
""",
unsafe_allow_html=True
)

if st.sidebar.button("🔴 Stop Tracker", key="stop_tracker_btn"):
    stop_capture()
    st.session_state.tracker_running = False
    
    st.sidebar.markdown("""
<div style="
background-color:#2C4A73;
padding:10px;
border-radius:4px;
font-weight:600;
color:black;">
Attention Tracker stopped.<br>
Refresh app to resume.
</div>
""", unsafe_allow_html=True)

# ====================== REAL-TIME ATTENTION PREDICTION ======================
WINDOW_SIZE = 1
DATA_FILE = Path("logs/attention_data.csv")

if "last_pred_time" not in st.session_state:
    st.session_state.last_pred_time = time.time()

st.markdown("<h1 style='color:black;font-weight:700;'> Live Attention Status</h1>",unsafe_allow_html=True)

if DATA_FILE.exists():

    df = pd.read_csv(DATA_FILE)

    if len(df) >= WINDOW_SIZE:

        window_df = df.tail(WINDOW_SIZE)

        face_ratio = window_df["face_detected"].mean()

        eye_open_ratio = (
            window_df["eye_gaze"] != "not_detected"
        ).mean()

        head_center_ratio = (
            window_df["eye_gaze"] == "center"
        ).mean()

        mouse_activity = int(
            window_df["mouse_clicks"].sum() > 0
        )

        mouse_movement_intensity = (
            window_df["mouse_x"].diff().abs().sum()
            + window_df["mouse_y"].diff().abs().sum()
        )

        mouse_click_rate = (
            window_df["mouse_clicks"].iloc[-1]
            / WINDOW_SIZE
        )

        gaze_focus_score = (
            window_df["eye_gaze"] == "center"
        ).mean()

        features = {
            "face_ratio": face_ratio,
            "eye_open_ratio": eye_open_ratio,
            "head_center_ratio": head_center_ratio,
            "mouse_activity": mouse_activity,
            "mouse_movement_intensity": mouse_movement_intensity,
            "mouse_click_rate": mouse_click_rate,
            "gaze_focus_score": gaze_focus_score,
            "attention_continuity": st.session_state.attention_continuity
        }

        prediction, confidence = predict_attention(features)

        st.session_state.attention_continuity = prediction

        if prediction == 1:
            st.success(
                f"✅ ATTENTIVE | Confidence: {confidence:.2f}"
            )
        else:
            st.error(
                f"⚠️ NOT ATTENTIVE | Confidence: {confidence:.2f}"
            )

    else:
        st.info("Collecting data… please wait.")

else:
    st.warning("Activity file not found.")

# ====================== SIDEBAR NAVIGATION ======================
# Navigation Heading
page = st.sidebar.radio(
    "📂 Navigation",
    ["Videos", "Notes", "Assessment", "Analytics"],
    key="sidebar_navigation"
)


# ====================== VIDEOS PAGE ======================
if page == "Videos":

    st.subheader(" 🎥 Learning Videos")

    video_path = Path(
        r"C:\Projects\CursorCam\CursorCam Project\data\Sample_Video_1.mp4"
    )

    if video_path.exists():

        video_bytes = video_path.read_bytes()

        # Center the video and reduce width
        col1, col2, col3 = st.columns([1, 5, 1])

        with col2:
            st.video(video_bytes)

    else:
        st.warning("Video file not found.")



# ====================== NOTES PAGE ======================
elif page == "Notes":

    st.subheader("📚 Study Notes")

    notes_path = Path(
        r"C:\Projects\CursorCam\CursorCam Project\data\Note File.txt"
    )

    if notes_path.exists():

        notes = notes_path.read_text(
            encoding="utf-8"
        )

        # Display notes inside a neat container
        st.text_area(
        "Notes",
        notes,
        height=300
)

    else:
        st.warning("Notes file not found.")


# ====================== TEST PAGE ======================
elif page == "Assessment":
    st.subheader("🧩 Assessment Section")
    st.write("Answer all questions and click **Submit** to view your score.")

    questions = [
        {"q": "Which of the following is a supervised learning algorithm?",
         "options": ["K-Means", "Decision Tree", "Apriori", "DBSCAN"],
         "answer": "Decision Tree"},

        {"q": "Which library is used for computer vision tasks in Python?",
         "options": ["Matplotlib", "OpenCV", "TensorFlow", "Numpy"],
         "answer": "OpenCV"},

        {"q": "Which metric is commonly used for classification accuracy?",
         "options": ["MSE", "Precision", "Recall", "Accuracy Score"],
         "answer": "Accuracy Score"},

        {"q": "What does CNN stand for?",
         "options": ["Convolutional Neural Network", "Central Neural Node",
                     "Convergent Network Node", "Computed Neural Network"],
         "answer": "Convolutional Neural Network"},

        {"q": "Which activation function outputs between 0 and 1?",
         "options": ["ReLU", "Sigmoid", "Tanh", "Softmax"],
         "answer": "Sigmoid"},

        {"q": "Which function is used to read CSV files in pandas?",
         "options": ["pd.read()", "pd.load_csv()", "pd.read_csv()", "pd.readfile()"],
         "answer": "pd.read_csv()"},

        {"q": "Which library is primarily used for deep learning?",
         "options": ["Scikit-learn", "Matplotlib", "TensorFlow", "NumPy"],
         "answer": "TensorFlow"},

        {"q": "In which OSI layer does IP operate?",
         "options": ["Transport", "Network", "Data Link", "Physical"],
         "answer": "Network"},

        {"q": "Which one is NOT a Python data type?",
         "options": ["Tuple", "List", "Set", "Dictionarys"],
         "answer": "Dictionarys"},

        {"q": "Which of the following is an unsupervised algorithm?",
         "options": ["Linear Regression", "K-Means", "Decision Tree", "Logistic Regression"],
         "answer": "K-Means"}
    ]

    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    for i, q in enumerate(questions):
        st.markdown(
f"""
<h2 style="
font-size:28px;
font-weight:700;
margin-bottom:5px;
">
Q{i+1}. {q['q']}
</h2>
""",
unsafe_allow_html=True
)
        choice = st.radio(
            "Select an option",
            q["options"],
            key=f"question_{i}",
            index=None
        )
        st.session_state.user_answers[i] = choice
        st.markdown(
"<div style='height:5px'></div>",
unsafe_allow_html=True
)

    if st.button("Submit Test", key="submit_test_btn"):
        if None in st.session_state.user_answers.values():
            st.error("⚠️ Please answer all questions.")
        else:
            st.session_state.submitted = True

    if st.session_state.submitted:
        score = 0
        st.subheader("📝 Results")

        for i, q in enumerate(questions):
            user_ans = st.session_state.user_answers[i]
            correct = q["answer"]

            if user_ans == correct:
                score += 1
                st.success(f"Q{i + 1}: Correct ✔️")
            else:
                st.error(f"Q{i + 1}: Wrong ❌ | Correct: {correct}")

        st.success(f"🎉 Final Score: {score} / {len(questions)}")


# ====================== ANALYTICS PAGE ======================
elif page == "Analytics":

    st.markdown("""
<h1 style="
font-size:34px;
font-weight:700;
color:black;
">
📈 Analytics Dashboard
</h1>
""", unsafe_allow_html=True)

    DATA_FILE = Path("logs/attention_data.csv")

    if not DATA_FILE.exists():

        st.warning("No attention data available yet.")

    else:

        df = pd.read_csv(DATA_FILE)

        if len(df) < 10:

            st.info("Not enough data to generate analytics.")

        else:

            # Attentive Frame Logic
            df["attentive_frame"] = (
                (df["face_detected"] == True) &
                (df["eye_gaze"] == "center")
            ).astype(int)

            # Group every 10 entries
            df["iteration_group"] = df.index // 10

            attention_by_iteration = (
                df.groupby("iteration_group")["attentive_frame"]
                .mean()
                .reset_index()
            )

            attention_by_iteration.rename(
                columns={
                    "attentive_frame": "attention_score"
                },
                inplace=True
            )

            st.markdown("<p style='color:#E91E63;font-weight:600;'>Average attention score calculated from every 10 interaction records.</p>",unsafe_allow_html=True)

            chart_data = attention_by_iteration.set_index(
                "iteration_group"
            )["attention_score"]

            # Smaller graph
            st.line_chart(
                chart_data,
                height=250
            )

            st.markdown("<p style='color:#2563EB;font-weight:600;'>1 = Fully Attentive | 0 = Not Attentive</p>",unsafe_allow_html=True)
