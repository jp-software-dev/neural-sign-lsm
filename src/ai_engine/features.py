# Preprocesamiento compartido de landmarks entre entrenamiento e inferencia.
# Convierte las coordenadas absolutas de la imagen en una representación invariante
# a la posición y al tamaño de la mano, de modo que el modelo aprenda la forma de la seña
# y no el lugar de la pantalla donde se capturó.
import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3

WRIST = 0


def normalize_landmarks(landmarks):
    # Acepta un vector plano de 63 valores o un arreglo (N, 63) y devuelve el mismo formato.
    arr = np.asarray(landmarks, dtype=np.float32)
    single = arr.ndim == 1
    pts = arr.reshape(-1, NUM_LANDMARKS, 3)

    # Traslada el origen a la muñeca
    pts = pts - pts[:, WRIST:WRIST + 1, :]

    # Escala por la envergadura de la mano: la mayor distancia (en x, y) de un punto a la muñeca.
    # Es estable aunque la palma apunte a la cámara, a diferencia de usar un solo nudillo.
    scale = np.linalg.norm(pts[:, :, :2], axis=2).max(axis=1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    pts = pts / scale[:, None, None]

    out = pts.reshape(-1, NUM_FEATURES)
    return out[0] if single else out
