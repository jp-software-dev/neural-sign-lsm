import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from src.config.settings import MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE

# ==========================================
# Índices de conexiones de la mano (MediaPipe)
# Cada tupla (A, B) dibuja una línea entre el landmark A y el B
# ==========================================
HAND_CONNECTIONS = [
    # Palma
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Pulgar
    (1, 2), (2, 3), (3, 4),
    # Índice
    (5, 6), (6, 7), (7, 8),
    # Medio
    (9, 10), (10, 11), (11, 12),
    # Anular
    (13, 14), (14, 15), (15, 16),
    # Meñique
    (17, 18), (18, 19), (19, 20),
]


class HandTracker:
    def __init__(self, ema_alpha=0.25):
        # Especificamos la ruta del modelo lite delegando la instanciación computacional al subsistema nativo de MediaPipe
        model_path = "hand_landmarker.task"
        base_options = BaseOptions(model_asset_path=model_path)

        # Inicializamos los umbrales estadísticos requeridos por la máquina de inferencia bloqueando detecciones sub-óptimas
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_TRACKING_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE
        )

        # Compilamos el modelo subyacente creando un puntero al detector TFLite con operaciones intrínsecas aceleradas
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

        # Definimos el coeficiente alpha ponderador decayente para el cálculo iterativo del Promedio Móvil Exponencial
        self.ema_alpha = ema_alpha
        # Buffer histórico de coordenadas suavizadas; None hasta recibir el primer fotograma válido
        self.prev_smoothed_landmarks = None
        self.consecutive_no_hand = 0

    # ------------------------------------------------------------------
    # Detección
    # ------------------------------------------------------------------

    def find_hands(self, frame):
        """Procesa un frame BGR y almacena los resultados internamente."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.results = self.detector.detect(mp_image)
        return self.results

    def process_frame(self, frame):
        """Alias de find_hands para mantener consistencia con data_gather.py."""
        return self.find_hands(frame)

    # ------------------------------------------------------------------
    # Landmarks
    # ------------------------------------------------------------------

    def get_landmarks(self, results=None, smooth=True):
        """
        Retorna una lista plana de 63 floats [x0,y0,z0, x1,y1,z1, ...]
        o [] si no hay mano detectada.
        """
        target_results = results if results is not None else self.results

        if not target_results or not target_results.hand_landmarks:
            self.prev_smoothed_landmarks = None
            self.consecutive_no_hand += 1
            return []

        self.consecutive_no_hand = 0
        hand = target_results.hand_landmarks[0]
        current_raw = [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]

        if not smooth:
            return current_raw

        if self.prev_smoothed_landmarks is None:
            self.prev_smoothed_landmarks = current_raw
            return current_raw

        smoothed = [
            self.ema_alpha * current_raw[i] + (1 - self.ema_alpha) * self.prev_smoothed_landmarks[i]
            for i in range(63)
        ]
        self.prev_smoothed_landmarks = smoothed
        return smoothed

    def reset_smoothing(self):
        """Reinicia el acumulador EMA (p. ej. al iniciar una nueva ronda de juego)."""
        self.prev_smoothed_landmarks = None

    # ------------------------------------------------------------------
    # Confianza de detección
    # FIX: La API de MediaPipe Tasks expone `handedness[0].score`, no
    #      `hand_presence`. El atributo anterior causaba una excepción
    #      silenciosa y retornaba siempre 0.0, deshabilitando el umbral.
    # ------------------------------------------------------------------

    def get_hand_confidence(self):
        """
        Retorna la confianza de detección de la mano [0.0 – 1.0].
        Usa handedness[0].score que sí existe en MediaPipe Tasks 0.10+.
        """
        if not self.results or not self.results.handedness:
            return 0.0
        try:
            # handedness es una lista de listas de Category; tomamos el score del primero
            return float(self.results.handedness[0][0].score)
        except (IndexError, AttributeError):
            return 0.0

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------

    def draw_landmarks(self, frame, smoothed_landmarks):
        """
        Dibuja los 21 landmarks Y las conexiones del esqueleto de la mano
        sobre el frame recibido (in-place).
        """
        if not smoothed_landmarks or len(smoothed_landmarks) != 63:
            return frame

        h, w = frame.shape[:2]

        # Convertimos la lista plana en array (21, 3) para indexación limpia
        pts = np.array(smoothed_landmarks).reshape(21, 3)
        pixel_pts = [(int(pts[i, 0] * w), int(pts[i, 1] * h)) for i in range(21)]

        # Dibujamos las conexiones del esqueleto en blanco semitransparente
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, pixel_pts[a], pixel_pts[b], (220, 220, 220), 1, cv2.LINE_AA)

        # Dibujamos los puntos articulares encima de las líneas
        for idx, (px, py) in enumerate(pixel_pts):
            # Punta de dedos más grandes para distinguirlas visualmente
            radius = 6 if idx in (4, 8, 12, 16, 20) else 4
            cv2.circle(frame, (px, py), radius, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius, (0, 180, 80), 1, cv2.LINE_AA)

        return frame

    def draw_hands(self, frame):
        """Calcula landmarks suavizados y los dibuja. Útil en data_gather."""
        smoothed = self.get_landmarks(smooth=True)
        return self.draw_landmarks(frame, smoothed)