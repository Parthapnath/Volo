# Volo: The Holistic Driver Co-Pilot

## Problem Statement

Road accidents are a global crisis, with human error accounting for over 90% of incidents. While traditional driver monitoring often focuses on singular issues like basic drowsiness, this approach is incomplete. Drivers face a spectrum of impairments – from **intoxication (alcohol)**, through **environmental factors (CO2-induced fatigue)**, to **emotional states like stress and road rage**. Current systems largely fail to address this holistic threat, treating drivers as problems to be surveilled rather than humans to be supported proactively.

## Volo's Solution: A Multi-Layered, Empathic Approach

**Volo** is an intelligent, multi-layered co-pilot that transitions driver monitoring from reactive surveillance to proactive, empathic safety. We fuse various data streams to build a holistic understanding of the driver's state and intervene intelligently.

### Our Core Pillars:

1.  **Pre-Drive Compliance & Safety Lockout:**
    * **Alcohol Detection:** An integrated sensor ensures the driver is sober *before* the car starts, providing a critical initial safety gate.

2.  **Passive Environmental Monitoring:**
    * **CO2 Level Monitoring:** In-cabin sensors detect elevated CO2, a known contributor to drowsiness. Volo proactively adjusts HVAC to cycle fresh air, preventing fatigue before it sets in.

3.  **Active Driver State Monitoring (The Empathic Co-Pilot):**
    * **AI-Powered Fusion:** This is Volo's innovative core. We combine:
        * **Visual Cues (Computer Vision):** Analyzing facial landmarks for drowsiness (eye closure, yawns), distraction, and emotional expressions.
        * **Physical Signatures (Vehicle Data):** Monitoring subtle interactions with the car, such as steering wheel grip pressure (simulated/CAN bus), sudden accelerations/decelerations, and sharp turns.
    * **Holistic Driver State Analysis (HDSA):** Our AI fusion model processes these inputs to differentiate between distinct states like:
        * **Fatigue:** (e.g., prolonged eye closure, yawning, relaxed grip).
        * **Agitation/Stress/Anger:** (e.g., tense facial expressions, tight grip, aggressive driving patterns).
        * **Distraction:** (e.g., looking away from the road, using a phone).

### Intelligent Interventions:

Volo's actions are tailored and human-centric, designed to de-escalate or alert without causing further distraction:

* **If Alcohol Detected:** Vehicle immobilization, "Do Not Drive" alert.
* **If High CO2:** Automatic fresh air circulation.
* **If Fatigue:** Alerting ambiance (light change), voice suggestions for rest stops, haptic alerts (e.g., electric impulse bracelet).
* **If Agitation/Stress:** Calming ambiance (light change), subtle audio modulation, haptic breathing guidance.

## Novelty & Differentiated Approach

Most existing solutions rely solely on image processing for basic drowsiness. Volo's novelty lies in its **three-pronged, fused approach**:

1.  **Multi-Modal Data Fusion:** Combining computer vision, environmental data (CO2), compliance checks (alcohol), and physical driving signatures (grip, vehicle dynamics) for unprecedented accuracy.
2.  **Empathic AI:** Our unique ability to differentiate between *why* a driver is impaired (e.g., sleepy vs. angry) and trigger context-appropriate interventions.
3.  **Proactive & Preventative:** Moving beyond mere detection to anticipation and de-escalation, addressing a broader spectrum of accident causes.

## Implementation Plan (Post-Hackathon MVP)

Our hackathon prototype demonstrates the core AI fusion model for fatigue and emotion. The full implementation involves:

1.  **Data Acquisition:** Integrating real-time streams from in-cabin cameras, alcohol sensors, CO2 sensors, and vehicle CAN bus data.
2.  **Pre-processing & Feature Extraction:** Filtering sensor data and using AI/Deep Learning for image processing and feature extraction from all data streams.
3.  **Holistic Classification (AI & Fuzzy Logic):**
    * Our AI fusion model classifies the driver's state (Calm, Fatigue, Agitation, Distracted).
    * A Fuzzy Logic model further refines these classifications by evaluating various criteria (e.g., sudden accelerations, sharp turns, heart rate from steering wheel) against context-aware thresholds (e.g., road type) to characterize driving style and impairment levels.
4.  **Intelligent Intervention System:** Activating the appropriate, differentiated intervention (HMI changes, vehicle controls) based on the classified state.

## Usage (Hackathon Prototype)

Our current prototype demonstrates the computer vision core for drowsiness and emotion recognition.

```bash
git clone [https://github.com/parthapnath/ai-driver-safety.git](https://github.com/parthapnath/ai-driver-safety.git)
```

**Setup:**

1.  **Download Shape Predictor:**
    Download `shape_predictor_68_face_landmarks.dat` from [dlib's GitHub](https://www.google.com/search?q=https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and place it in the main project folder.
2.  **Install Dependencies:**
    ```bash
    pip install opencv-python dlib scipy imutils playsound==1.2.2 tensorflow numpy keras
    ```
    *(Note: `tensorflow` includes `keras`. `playsound` version 1.2.2 is recommended for stability.)*

---

### **1. Face Landmarks:**

Utilizes dlib's facial landmark predictor for real-time face analysis.

---

### **2. Drowsiness Detection (Eye Blinking & Closure):**

Detects eye blinks and prolonged eye closure. Triggers an alarm (`alarm.wav`) for potential drowsiness.

```bash
python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
```

*(Based on `drowsiness_detection.py` in the original repository, adapted for `detect_drowsiness.py` to match provided code files for demo guidance)*

---

### **3. Activity Recognition: Yawning:**

Analyzes facial landmarks of the mouth to detect yawning.

```bash
python yawn.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

*(Requires the `shape_predictor_68_face_landmarks.dat` file)*

---

### **4. Mood/Emotion Recognition:**

Utilizes a trained TensorFlow/Keras model to classify real-time mood/emotion (e.g., Neutral, Angry). This demonstrates Volo's capability to detect agitation.

```bash
python mood_recognition.py
```

*(Requires `model.h5` and `haarcascade_frontalface_default.xml` files)*