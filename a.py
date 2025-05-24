
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
from playsound import playsound
import threading
from twilio.rest import Client
import base64

# ========== CONFIGURATION ==========
TWILIO_SID = '....'  
TWILIO_AUTH = '....'  
FROM_PHONE = '....'       
TO_PHONE = '....'         
ALARM_FILE = 'siren-alert-96052.mp3'
MODEL_PATH = 'output/pose_classifier.pkl'

BUTTON_BG_IMAGE = 'background.jpeg' 

# ========== INITIALIZATION ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
model = joblib.load(MODEL_PATH)
alarm_thread = None
stop_alarm_flag = threading.Event()

activity_log = []
fall_log = []

# ========== FUNCTIONS ==========
def detect_fall(landmarks):
    try:
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        sh_y = (l_sh.y + r_sh.y) / 2
        hip_y = (l_hip.y + r_hip.y) / 2
        return abs(sh_y - hip_y) < 0.15
    except:
        return False

def send_sms_alert(message):
    client = Client(TWILIO_SID, TWILIO_AUTH)
    client.messages.create(body=message, from_=FROM_PHONE, to=TO_PHONE)

def play_alarm():
    stop_alarm_flag.clear()
    while not stop_alarm_flag.is_set():
        playsound(ALARM_FILE)
        time.sleep(1)

def stop_alarm():
    stop_alarm_flag.set()

def get_pose_color(label):
    colors = {
        'Standing': (0, 255, 0),
        'Sitting': (255, 255, 0),
        'Lying': (255, 0, 0),
        'Unknown': (128, 128, 128)
    }
    return colors.get(label, (255, 255, 255))

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_button_style():
    # Button background image encoded as base64
    encoded = get_base64_of_bin_file(BUTTON_BG_IMAGE)
    st.markdown(f"""
        <style>
        div.stButton > button {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            color: white;
            font-weight: bold;
            height: 40px;
            border-radius: 10px;
            border: none;
        }}
        div.stButton > button:hover {{
            filter: brightness(85%);
            cursor: pointer;
        }}
        </style>
    """, unsafe_allow_html=True)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Fall Detection Dashboard", layout="wide")

# Remove background image - do nothing here

set_button_style()

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Fall Log", "About"])

# Content based on sidebar selection
if selection == "Home":
    st.title("📹 Real-Time Pose Detection & Fall Alert System")

    # Buttons for controlling detection & logs (styled buttons)
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        start_button = st.button("Start Detection")
    with col2:
        stop_button = st.button("Stop Detection")
    with col3:
        show_fall_log = st.button("Show Fall Log")

    # Video & status columns
    colA, colB = st.columns([2,1])
    video_placeholder = colA.empty()
    status_placeholder = colB.empty()
    pose_placeholder = colB.empty()
    fall_log_placeholder = colB.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        fall_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            label = "Unknown"
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                for lm in landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                features = np.array(keypoints).flatten()
                if features.shape[0] == 132:
                    label = model.predict([features])[0]

                color = get_pose_color(label)
                cv2.putText(frame, f'Pose: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                is_fall = detect_fall(landmarks)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if is_fall and not fall_detected:
                    fall_detected = True
                    fall_log.append({"time": timestamp, "event": "Fall Detected"})
                    status_placeholder.error("🚨 Fall Detected!")
                    pose_placeholder.warning(f"Pose: {label}")
                    send_sms_alert("Fall detected! Immediate attention needed.")
                    if not (alarm_thread and alarm_thread.is_alive()):
                        alarm_thread = threading.Thread(target=play_alarm)
                        alarm_thread.start()

                elif not is_fall and fall_detected:
                    fall_detected = False
                    fall_log.append({"time": timestamp, "event": "Returned to Normal"})
                    status_placeholder.success("✅ Normal")
                    pose_placeholder.info(f"Pose: {label}")
                    send_sms_alert("Person has returned to normal pose.")
                    stop_alarm()

            video_placeholder.image(frame, channels="BGR")
            time.sleep(0.03)

        cap.release()

    if show_fall_log:
        st.subheader("📝 Fall Log")
        fall_df = pd.DataFrame(fall_log)
        fall_log_placeholder.dataframe(fall_df)

        csv_fall = fall_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Fall Log", csv_fall, "fall_log.csv", "text/csv")

    if stop_button:
        stop_alarm()
        st.success("Detection stopped.")

elif selection == "Fall Log":
    st.title("📝 Fall Log")
    fall_df = pd.DataFrame(fall_log)
    st.dataframe(fall_df)

    csv_fall = fall_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Fall Log", csv_fall, "fall_log.csv", "text/csv")

elif selection == "About":
    st.title("About This App")
    st.write("""
    This is a Real-Time Pose Detection & Fall Alert System using MediaPipe and machine learning.
    Alerts are sent via Twilio SMS.
    Developed with Streamlit.
    """)

