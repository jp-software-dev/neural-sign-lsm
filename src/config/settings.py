import os

# Directorio raíz del proyecto
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))

# Rutas de datos y modelos
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LANDMARKS_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "Landmarks.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
AI_MODEL_PATH = os.path.join(MODELS_DIR, "lsm_model.h5")
LABELS_JSON_PATH = os.path.join(DATA_PROCESSED_DIR, "Labels.json")

# Rutas de registros y logs
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Parámetros de la cámara y procesamiento
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS_LIMIT = 30

# Parámetros de IA y rastreo
MP_DETECTION_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE = 0.5
AI_CONFIDENCE_THRESHOLD = 0.85

# Interfaz gráfica de colores en formato BGR
COLOR_SUCCESS = (0, 255, 5)
COLOR_DANGER = (0, 0, 255) 
COLOR_INFO = (255, 255, 0)