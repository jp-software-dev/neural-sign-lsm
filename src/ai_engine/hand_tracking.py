# Encapsula la lógica de detección y seguimiento de manos utilizando MediaPipe.
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from src.config.settings import MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE

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
    # Clase para detectar landmarks de una mano en un frame de video,
    # aplicar un filtro de suavizado y dibujar los resultados.
    def __init__(self, ema_alpha=0.25):
        # --- Configuración del Detector de MediaPipe ---
        model_path = "hand_landmarker.task"
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_TRACKING_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

        # --- Configuración del Filtro de Suavizado (EMA) ---
        self.ema_alpha = ema_alpha
        self.prev_smoothed_landmarks = None
        self.consecutive_no_hand = 0

    def find_hands(self, frame):
        # Procesa un frame de video para detectar manos.
        # Los resultados se guardan en `self.results`.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.results = self.detector.detect(mp_image)
        return self.results

    def process_frame(self, frame):
        # Alias de `find_hands` para compatibilidad con otros módulos.
        return self.find_hands(frame)

    # Landmarks
    def get_landmarks(self, results=None, smooth=True):
        # Extrae las coordenadas de los landmarks de la mano detectada.
        # Aplica un filtro de suavizado (Media Móvil Exponencial) si está activado.
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
        # Reinicia el filtro de suavizado, útil al cambiar de modo o juego.
        self.prev_smoothed_landmarks = None

    def get_hand_confidence(self):
        # Retorna la puntuación de confianza de que una mano está presente.
        if not self.results or not self.results.handedness:
            return 0.0
        try:
            return float(self.results.handedness[0][0].score)
        except (IndexError, AttributeError):
            return 0.0
        
    # Dibujo
    def draw_landmarks(self, frame, smoothed_landmarks):
        # Dibuja el esqueleto y los puntos de la mano sobre un frame.
        if not smoothed_landmarks or len(smoothed_landmarks) != 63:
            return frame

        h, w = frame.shape[:2]

        # Convertimos la lista plana en array (21, 3) para indexación limpia
        pts = np.array(smoothed_landmarks).reshape(21, 3)
        pixel_pts = [(int(pts[i, 0] * w), int(pts[i, 1] * h)) for i in range(21)]

        # Dibuja las conexiones del esqueleto
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, pixel_pts[a], pixel_pts[b], (220, 220, 220), 1, cv2.LINE_AA)

        # Dibuja los puntos de los landmarks
        for idx, (px, py) in enumerate(pixel_pts):
            radius = 6 if idx in (4, 8, 12, 16, 20) else 4
            cv2.circle(frame, (px, py), radius, (0, 255, 120), -1)

        return frame

    def draw_hands(self, frame):
        # Método de conveniencia que detecta, suaviza y dibuja en un solo paso.
        smoothed = self.get_landmarks(smooth=True)
        return self.draw_landmarks(frame, smoothed)