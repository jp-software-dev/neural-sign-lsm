# Encapsula la lógica de detección y seguimiento de manos utilizando MediaPipe.
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import threading
from src.config.settings import MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE, HAND_LANDMARKER_PATH

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
    def __init__(self, ema_alpha=0.2):
        self.latest_results = None
        self.lock = threading.Lock()

        def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            with self.lock:
                self.latest_results = result

        base_options = BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_TRACKING_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            result_callback=result_callback,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # --- Configuración del Filtro de Suavizado (EMA) ---
        self.base_ema_alpha = ema_alpha
        self.prev_smoothed_landmarks = None
        self.prev_raw_landmarks = None
        self.last_dynamic_alpha = ema_alpha

        # Último resultado ya suavizado, para no aplicar el EMA dos veces al mismo resultado
        self._last_processed_result = None
        self._last_output = []
        self.last_result_is_new = False
        self._last_timestamp_ms = -1

    def process_frame_async(self, frame, timestamp_ms: int):
        # Procesa un frame de video para detectar manos de forma asíncrona.
        # MediaPipe exige timestamps estrictamente crecientes en modo LIVE_STREAM
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)

    # Landmarks
    def get_landmarks(self, smooth=True):
        # Extrae las coordenadas de los landmarks de la mano detectada.
        # Aplica un filtro de suavizado (Media Móvil Exponencial) si está activado.
        with self.lock:
            target_results = self.latest_results

        # Si el detector aún no entrega un resultado nuevo, se devuelve el último cálculo
        if target_results is not None and target_results is self._last_processed_result and smooth:
            self.last_result_is_new = False
            return list(self._last_output)
        self._last_processed_result = target_results
        self.last_result_is_new = True

        if not target_results or not target_results.hand_landmarks:
            self.prev_smoothed_landmarks = None
            self.prev_raw_landmarks = None
            self._last_output = []
            return []

        hand = target_results.hand_landmarks[0]
        current_raw = [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]

        if not smooth:
            self.prev_raw_landmarks = current_raw
            return current_raw

        # --- Cálculo Dinámico de Alpha ---
        dynamic_alpha = self.base_ema_alpha
        if self.prev_raw_landmarks is not None:
            # Calcula la velocidad basándose en el desplazamiento del landmark de la muñeca (índice 0)
            wrist_current = np.array(current_raw[0:3])
            wrist_prev = np.array(self.prev_raw_landmarks[0:3])
            velocity = np.linalg.norm(wrist_current - wrist_prev)

            # Mapea la velocidad a alpha. Estos valores se pueden ajustar.
            # Una mayor velocidad resulta en un alpha más alto (menos suavizado, más reactividad).
            velocity_scale = 15.0  # Factor para amplificar el efecto de la velocidad
            min_alpha = self.base_ema_alpha
            max_alpha = 0.9  # Límite superior para alpha para prevenir jitter excesivo

            dynamic_alpha = min(min_alpha + velocity * velocity_scale, max_alpha)

        self.last_dynamic_alpha = dynamic_alpha

        if self.prev_smoothed_landmarks is None:
            self.prev_smoothed_landmarks = current_raw
            self.prev_raw_landmarks = current_raw
            self._last_output = current_raw
            return list(current_raw)

        smoothed = [
            dynamic_alpha * current_raw[i] + (1 - dynamic_alpha) * self.prev_smoothed_landmarks[i]
            for i in range(63)
        ]
        self.prev_smoothed_landmarks = smoothed
        self.prev_raw_landmarks = current_raw
        self._last_output = smoothed
        return list(smoothed)

    def reset_smoothing(self):
        # Reinicia el filtro de suavizado, útil al cambiar de modo o juego.
        self.prev_smoothed_landmarks = None
        self.prev_raw_landmarks = None
        self._last_processed_result = None
        self._last_output = []

    def get_last_dynamic_alpha(self):
        return self.last_dynamic_alpha

    def get_hand_confidence(self):
        # Retorna la puntuación de confianza de que una mano está presente.
        with self.lock:
            target_results = self.latest_results
        if not target_results or not target_results.handedness:
            return 0.0
        try:
            return float(target_results.handedness[0][0].score)
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
