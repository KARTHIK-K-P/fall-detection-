# 📹 Real-Time Fall Detection & Alert System

This Streamlit app uses **MediaPipe**, **OpenCV**, and a pre-trained ML model to detect human poses in real time and identify potential falls. If a fall is detected, it can play an audible alarm and send an SMS alert using Twilio.

---

## 🚀 Features

- Real-time video feed from webcam.
- Pose classification using a pre-trained model.
- Fall detection based on pose and position.
- Audio alarm using `playsound`.
- SMS alerts via Twilio API.
- Interactive UI with activity logging and CSV export.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fall-detection-streamlit.git
cd fall-detection-streamlit


streamlit run a.py 
