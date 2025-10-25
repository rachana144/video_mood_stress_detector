import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ===============================
# 🌈 APP CONFIGURATION
# ===============================
st.set_page_config(page_title="AIstra - Emotion Detector", layout="wide")

# 💅 Enhanced colorful theme with robust BLACK text for sidebar controls
st.markdown("""
    <style>
        /* Background gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #C3F8FF 0%, #D7C4F3 50%, #FFE6E6 100%);
            background-attachment: fixed;
            color: #1A237E;
        }

        /* Main content area */
        .main > div {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }

        /* Title and subtitle */
        .title {
            text-align: center;
            font-size: 58px;
            font-weight: 900;
            color: #6A1B9A;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            font-size: 22px;
            color: #303F9F;
            font-weight: 500;
            margin-bottom: 30px;
        }

        /* Sidebar gradient and header */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #E1BEE7 0%, #BBDEFB 100%);
            color: #1A237E;
            border-right: 2px solid rgba(106,27,154,0.2);
        }
        .sidebar-content h2 {
            color: #6A1B9A;
            text-align: center;
        }

        /* Buttons */
        .stButton>button {
            background-color: #D1C4E9;
            color: #4A148C;
            border-radius: 10px;
            border: 1px solid #4A148C;
            padding: 0.5em 1.2em;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #7E57C2;
            color: white;
            border: 1px solid white;
        }
        
        /* === BLACK TEXT FIXES FOR SIDEBAR CONTROLS === */

        /* 🖤 Checkbox label text in black (Refined Selector) */
        div[data-testid="stSidebar"] div[data-testid="stCheckbox"] label span {
             color: black !important;
             font-weight: 600;
        }
        
        /* Optional: Add a light background to the checkbox area for better contrast on the gradient */
        div[data-testid="stSidebar"] div[data-testid="stCheckbox"] {
            background-color: rgba(255,255,255,0.9);
            border-radius: 10px;
            padding: 6px 10px;
            margin-bottom: 10px; /* Space it out */
        }
        
        /* 🖤 Selectbox label text in black (Refined Selector) */
        div[data-testid="stSidebar"] label {
            color: black !important;
            font-weight: 600;
        }
        
        /* === END BLACK TEXT FIXES === */

        /* Info/warning boxes */
        div[data-testid="stStatusWidget"] div[role="alert"] {
            background-color: #E3F2FD;
            color: #1A237E;
            border-left: 5px solid #512DA8;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💫 AIstra</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Empathy-powered Emotion Detection 🎭</div>", unsafe_allow_html=True)
st.write("")

# ===============================
# 🎭 EMOJI CONFIGURATION
# ===============================
EMOJI_PATH = "emojis"
EMOJI_MAP = {
    "happy": "happy.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "surprise": "surprise.png",
    "neutral": "neutral.png",
    "fear": "fear.png",
    "disgust": "disgust.png",
    "No Face": "neutral.png"
}

# ===============================
# ⚙ SIDEBAR CONTROLS
# ===============================
st.sidebar.header("🪄 AIstra Controls")

source = st.sidebar.selectbox("🎥 Choose Input Source", ["Webcam", "Upload Video"])
# The label for this checkbox will be black due to the CSS fix
show_plot = st.sidebar.checkbox("📊 Show Emotion Timeline (Matplotlib)", True) 

st.sidebar.markdown("---")
start_btn = st.sidebar.button("🎥 Start Camera")
stop_btn = st.sidebar.button("⏹ Stop Camera")

# ===============================
# SESSION STATES
# ===============================
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None
if "running" not in st.session_state:
    st.session_state.running = False

# ===============================
# 📸 CAMERA / VIDEO LOGIC
# ===============================
if start_btn:
    st.session_state.running = True
    st.session_state.emotion_log = []
    st.session_state.last_emotion = None

if stop_btn:
    st.session_state.running = False

FRAME_WINDOW = st.empty()

if st.session_state.running:
    temp_path = None
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video_uploader")
        if uploaded_file is not None:
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_path)
        else:
            st.warning("📽 Please upload a video file to continue.")
            st.stop()

    st.info("🎥 Camera running... Click ⏹ Stop Camera to end.")
    frame_count = 0

    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame is None or frame.size == 0:
            continue

        if frame_count % 5 == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
            except Exception:
                dominant_emotion = "No Face"

            st.session_state.emotion_log.append((time.time(), dominant_emotion))
            st.session_state.last_emotion = dominant_emotion
        else:
            dominant_emotion = st.session_state.last_emotion if st.session_state.last_emotion else "No Face"

        # 🧩 Emoji Overlay
        emoji_key = dominant_emotion if dominant_emotion != "No Face" else "neutral"
        emoji_file = os.path.join(EMOJI_PATH, EMOJI_MAP.get(emoji_key, "neutral.png"))

        if os.path.exists(emoji_file):
            emoji_img = cv2.imread(emoji_file, cv2.IMREAD_UNCHANGED)
            if emoji_img is not None and emoji_img.shape[2] == 4:
                h, w = emoji_img.shape[:2]
                scale = 0.08
                new_w, new_h = int(w * scale), int(h * scale)
                emoji_resized = cv2.resize(emoji_img, (new_w, new_h))
                x_offset, y_offset = frame.shape[1] - new_w - 10, 10
                
                # Check bounds before overlay
                if y_offset + new_h <= frame.shape[0] and x_offset + new_w <= frame.shape[1]:
                    alpha = emoji_resized[:, :, 3] / 255.0
                    for c in range(3):
                        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                            alpha * emoji_resized[:, :, c] +
                            (1 - alpha) * frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                        )

        # Black text for video overlay for visibility
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.15)

    cap.release()
    if source == "Upload Video" and temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
    st.session_state.running = False

# ===============================
# 📊 EMOTION TREND (MATPLOTLIB)
# ===============================
if show_plot and len(st.session_state.emotion_log) > 1:
    st.subheader("📈 Emotion Timeline")
    times = [t - st.session_state.emotion_log[0][0] for t, _ in st.session_state.emotion_log]
    emotions = [e for _, e in st.session_state.emotion_log]
    plot_emotions = [e for e in emotions if e != "No Face"]
    unique_emotions = sorted(list(set(plot_emotions)))
    emotion_to_num = {e: i for i, e in enumerate(unique_emotions)}
    emotion_nums = [emotion_to_num.get(e, -1) for e in emotions]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 4))
    valid_indices = [i for i, num in enumerate(emotion_nums) if num != -1]
    valid_times = [times[i] for i in valid_indices]
    valid_emotion_nums = [emotion_nums[i] for i in valid_indices]
    ax.plot(valid_times, valid_emotion_nums, marker='o', linestyle='-', color='#7E57C2', linewidth=2)
    ax.set_yticks(list(emotion_to_num.values()))
    ax.set_yticklabels(list(emotion_to_num.keys()))
    ax.set_title("Emotion Trend Over Time", fontsize=16, color='#1A237E')
    ax.set_xlabel("Time (s)", color='#1A237E')
    ax.set_ylabel("Emotion", color='#1A237E')
    ax.grid(True, linestyle='--', alpha=0.6, color='gray')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    st.pyplot(fig)
    plt.close(fig)
elif not st.session_state.running:
    st.info("👆 Click '🎥 Start Camera' to begin capturing emotions.")
