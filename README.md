# NeuralSign-LSM

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

A Computer Vision and Deep Learning engine developed in Python. It maps the 21 joints of the hand in real time to translate the Mexican Sign Language (LSM) alphabet into text and voice. The model is trained on a custom dataset of **over 40,000 images** (41,753 webcam frames, ~1,500 per letter), and the app offers a Free Mode for real-time translation and a Gamified Learning Module for interactive practice.

## ✨ Features

* **Deep Learning Model:** A dense neural network (TensorFlow/Keras) that classifies the complete **A–Z alphabet** from MediaPipe hand landmarks, reaching **99.4% accuracy on recording sessions it never saw during training**.
* **Position and Size Invariant:** Landmarks are normalized relative to the wrist and the hand span, so recognition works anywhere in the frame and at any distance from the camera.
* **Free Mode (Real-Time Translation):** Adaptive smoothing (dynamic EMA) plus a stability check: a sign is confirmed only after it is held for several consecutive frames above a confidence threshold.
* **Gamified Mini-Game Module:** A hangman-style spelling game with three difficulty levels, streak multipliers, time bonuses, pause and a Top 3 high-score board.
* **Voice Assistant Integration:** An offline Text-to-Speech engine (pyttsx3) running on a background thread, so speech never blocks the video loop.
* **Fail-Safe Architecture:** Centralized error logging and a modular codebase that cleanly separates the AI engine from the user interface.

## 🛠 Technologies Used

* **Core Language:** Python 3.12
* **Computer Vision:** OpenCV, MediaPipe (Hand Landmarker)
* **Machine Learning:** TensorFlow / Keras, scikit-learn, NumPy, Pandas
* **Accessibility & Audio:** pyttsx3 (Text-to-Speech), pygame (sound effects, optional)

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jp-software-dev/neural-sign-lsm.git
   cd neural-sign-lsm
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application (the trained model is included in `models/`):
   ```bash
   python src/main.py
   ```

A webcam is required. Optional sound effects are loaded from `assets/sounds/` (`tick.wav`, `finish.wav`, `correct.wav`, `incorrect.wav`); without them the app runs silently.

## 🎮 Controls

| Key | Action |
|---|---|
| `1` | Free Mode (real-time translation) |
| `2` | Game Mode → choose difficulty `1` Easy, `2` Medium, `3` Hard, `4` Random, `5` Back |
| `3` | High scores (`B` to delete, `S` to confirm) |
| `P` | Pause / resume during a game |
| `4` / `Esc` | Exit |

To repeat the same letter in Free Mode (e.g. "LL"), lower your hand briefly between signs. In Game Mode, double letters (e.g. the "RR" in *PERRO*) are accepted by holding the sign again.

## 🧠 Training Your Own Model

1. **Capture data** (run from the project root):
   ```bash
   python -m src.ai_engine.data_gather
   ```
   Press a lowercase letter (`a`–`z`) to choose the sign, then `C` for one sample, `V` for a burst of 500 samples, `B` to record a 30-frame motion sequence and `Q` to quit. Samples are appended to `data/processed/landmarks.csv`.

2. **Train**:
   ```bash
   python -m src.ai_engine.trainer --mode static
   ```
   The trainer validates on the **most recent recording session of each letter** (data the model never sees during training), writes the report to `models/training_report_static.txt`, and then retrains on the full dataset for the best number of epochs. An experimental sequence model is available with `--mode lstm` (saved separately as `models/lsm_model_lstm.h5`; enable it in the app with `USE_LSTM_MODEL = True` in `src/config/settings.py`).

## 📂 Project Structure

```
NeuralSign-LSM/
├── hand_landmarker.task      # MediaPipe hand landmark model
├── models/                   # Trained model, labels, scaler and training report
├── data/processed/           # landmarks.csv (training dataset, not versioned)
└── src/
    ├── main.py               # Application loop, state machine and UI
    ├── config/settings.py    # Paths and hyperparameters
    ├── ai_engine/
    │   ├── hand_tracking.py  # MediaPipe tracking + adaptive smoothing
    │   ├── features.py       # Landmark normalization shared by training and inference
    │   ├── data_gather.py    # Dataset capture tool
    │   └── trainer.py        # Training, validation and reports
    ├── modules/              # Game logic, word bank, voice, sounds, high scores
    └── utils/logger.py       # Error logging (logs/app_errors.log)
```

## 🗺 Roadmap

* Recognition of full words and formal phrases.
* Motion-based recognition for dynamic letters (J, K, Ñ, Q, X, Z) with the LSTM model.
* Support for the letter Ñ.

## 👤 Author

Juan Pablo Mercado Arizmendi
