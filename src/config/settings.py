# Archivo de configuración central para constantes, rutas y hiperparámetros.
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SEQUENCES_DIR = os.path.join(DATA_DIR, "sequences")
REFERENCE_IMAGES_DIR = os.path.join(DATA_DIR, "reference_images")
LANDMARKS_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "landmarks.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
SOUNDS_DIR = os.path.join(PROJECT_ROOT, "assets", "sounds")
HAND_LANDMARKER_PATH = os.path.join(PROJECT_ROOT, "hand_landmarker.task")

AI_MODEL_PATH = os.path.join(MODELS_DIR, "lsm_model.h5")
LABELS_JSON_PATH = os.path.join(MODELS_DIR, "labels.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Artefactos del modelo de secuencias (LSTM), separados para no sobrescribir el modelo estático
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lsm_model_lstm.h5")
LSTM_LABELS_JSON_PATH = os.path.join(MODELS_DIR, "labels_lstm.json")
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_lstm.pkl")

# Si es True, la aplicación carga el modelo LSTM en lugar del estático
USE_LSTM_MODEL = False

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS_LIMIT = 30

MP_DETECTION_CONFIDENCE = 0.8
MP_TRACKING_CONFIDENCE = 0.7

AI_CONFIDENCE_THRESHOLD = 0.85
STABLE_FRAMES_REQUIRED = 5

SEQUENCE_LENGTH = 30

COLOR_SUCCESS = (0, 255, 0)
COLOR_DANGER = (0, 0, 255)
COLOR_INFO = (255, 255, 0)
