# NeuralSign-LSM

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

An advanced Computer Vision and Deep Learning engine developed in Python. It maps 21 articular nodes of the hand in real-time to accurately translate Mexican Sign Language (LSM) into both text and voice, featuring a gamified learning module for interactive practice.

## Features

* **Real-Time Translation:** Highly accurate hand tracking utilizing neural networks to dynamically classify the complete A-Z alphabet and formal phrases.
* **Voice Assistant Integration:** A built-in, offline Text-to-Speech (TTS) engine that instantly vocalizes detected signs with zero latency.
* **Gamified Learning Mode:** An interactive, hangman-style game where users spell random words using physical sign language to progress and build fluency.
* **Fail-Safe Architecture:** A robust automated internal logging system and a modular codebase that cleanly separates the AI engine from the user interface.

## Technologies Used

* **Core Language:** Python 3.x
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** TensorFlow / Keras, NumPy, Pandas
* **Accessibility:** pyttsx3 (Text-to-Speech)

## Installation & Setup

Follow these steps to deploy the project locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/jp-software-dev/NeuralSign-LSM.git](https://github.com/jp-software-dev/NeuralSign-LSM.git)
