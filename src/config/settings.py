import os

# Resolvemos la ruta absoluta del directorio raiz escalando tres niveles en el arbol de directorios del sistema de archivos local
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Definimos las rutas estaticas para la persistencia del dataset procesado de coordenadas espaciales extraidas
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LANDMARKS_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "landmarks.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Mapeamos los artefactos serializados resultantes del pipeline de entrenamiento para su carga dinamica durante la inferencia
AI_MODEL_PATH = os.path.join(MODELS_DIR, "lsm_model.h5")
LABELS_JSON_PATH = os.path.join(MODELS_DIR, "labels.json")
# Especificamos la ruta del objeto StandardScaler para garantizar que la distribucion estadistica de inferencia coincida con la de entrenamiento
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl") 

# Configuramos el directorio centralizado para el volcado de trazas de ejecucion y telemetria de errores del sistema
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Parametrizamos la resolucion matricial del hardware de captura de video para estandarizar el tensor de entrada espacial
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS_LIMIT = 30

# Definimos los hiperparametros de umbral para el motor de inferencia de topologia de MediaPipe minimizando falsos positivos
MP_DETECTION_CONFIDENCE = 0.8
MP_TRACKING_CONFIDENCE = 0.7
# Establecemos el limite de activacion softmax requerido para confirmar la prediccion final de la red neuronal densa
AI_CONFIDENCE_THRESHOLD = 0.85

# Dimensionamos la ventana deslizante temporal requerida para procesar la secuencia de fotogramas en la arquitectura recurrente LSTM
SEQUENCE_LENGTH = 30 

# Asignamos constantes vectoriales en el espacio de color BGR nativo de OpenCV para la renderizacion de componentes de interfaz
COLOR_SUCCESS = (0, 255, 0)
COLOR_DANGER = (0, 0, 255) 
COLOR_INFO = (255, 255, 0)