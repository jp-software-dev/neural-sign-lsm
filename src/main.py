"""
main.py
-------
Punto de entrada principal de NeuralSign-LSM.

Mejoras respecto a la versión anterior:
  - Control real de FPS (evita quemar CPU/GPU en predicción continua)
  - Pantalla de selección de dificultad antes del juego
  - HUD mejorado: racha, palabras completadas, barra de tiempo
  - Progress display de la palabra (letras acertadas / pendientes)
  - Pantalla de resumen al terminar la sesión
"""

import cv2
import numpy as np
import json
import sys
import os
import pickle
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from src.config.settings import (
    AI_MODEL_PATH, LABELS_JSON_PATH, SCALER_PATH,
    SEQUENCE_LENGTH, AI_CONFIDENCE_THRESHOLD,
    COLOR_SUCCESS, COLOR_DANGER, COLOR_INFO, FPS_LIMIT,
)
from src.ai_engine.hand_tracking import HandTracker
from src.modules import VoiceAssistant, SignGame


# ── Constantes de UI ────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_BG_COLOR = (10, 10, 10)
STREAK_COLOR = (0, 215, 255)   # dorado
WORD_COLOR = (255, 255, 80)    # amarillo suave
PROGRESS_COLOR = (180, 255, 180)

DIFFICULTY_KEYS = {
    ord('1'): "facil",
    ord('2'): "medio",
    ord('3'): "dificil",
}


# ── Helpers de dibujo ───────────────────────────────────────────────────

def draw_hud(frame, status_text, color, game):
    """Dibuja el HUD principal encima del frame."""
    h, w = frame.shape[:2]

    # Fondo semitransparente superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Línea 1: estado de predicción
    cv2.putText(frame, status_text, (15, 35), FONT, 0.8, color, 2, cv2.LINE_AA)

    if game.game_active:
        # Línea 2: palabra objetivo con progreso
        progress = game.progress_display
        cv2.putText(frame, f"Palabra: {progress}", (15, 68), FONT, 0.8, WORD_COLOR, 2, cv2.LINE_AA)

        # Línea 3: puntos, racha, palabras, tiempo
        hud_right = (
            f"Pts:{game.score}  "
            f"Racha:{game.streak}x  "
            f"Palabras:{game.words_completed}  "
            f"Tiempo:{game.time_left}s"
        )
        cv2.putText(frame, hud_right, (15, 100), FONT, 0.55, STREAK_COLOR, 1, cv2.LINE_AA)

        # Barra de tiempo en la parte inferior
        bar_w = int(w * game.time_left / game.time_limit)
        bar_color = COLOR_SUCCESS if game.time_left > 20 else COLOR_DANGER
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), bar_color, -1)

    # Instrucciones en la parte inferior
    hint = "J:Jugar[1/2/3] | K:Salir"
    cv2.putText(frame, hint, (15, h - 16), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_summary(frame, game):
    """Overlay de resumen al terminar una sesión."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 4, h // 4), (w * 3 // 4, h * 3 // 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    lines = [
        "SESION TERMINADA",
        f"Puntuacion: {game.score} pts",
        f"Palabras:   {game.words_completed}",
        f"Mejor racha: {game.streak}x",
        "",
        "J: Nueva sesion | K: Salir",
    ]
    for i, line in enumerate(lines):
        y = h // 4 + 40 + i * 34
        color = WORD_COLOR if i == 0 else (220, 220, 220)
        scale = 0.85 if i == 0 else 0.7
        cv2.putText(frame, line, (w // 4 + 20, y), FONT, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    tracker = HandTracker(ema_alpha=0.25)
    voice = VoiceAssistant()

    # Carga de artefactos de la IA
    try:
        model = load_model(AI_MODEL_PATH)

        with open(LABELS_JSON_PATH, 'r') as f:
            labels_dict = json.load(f)
        labels = {int(k): v for k, v in labels_dict.items()}

        available_letters = list(labels.values())

        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        print(f"Sistema cargado. Letras disponibles: {available_letters}")
    except Exception as e:
        print(f"ERROR FATAL: No se pudo cargar el modelo o artefactos. {e}")
        print("Corre primero trainer.py para generar los archivos.")
        return

    is_lstm = len(model.input_shape) == 3
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) if is_lstm else None
    print(f"Modo de inferencia: {'LSTM' if is_lstm else 'Estático'}")

    # Instanciamos el motor del juego con dificultad inicial "facil"
    game = SignGame(available_letters, difficulty="facil")

    # Cámara
    cap = cv2.VideoCapture(0)
    window_name = "NeuralSign-LSM - Traductor de Lengua de Señas Mexicana"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    last_prediction = ""
    show_summary = False
    voice.speak("Sistema iniciado. Presiona J para jugar o K para salir.")

    # Control de FPS
    frame_interval = 1.0 / FPS_LIMIT
    last_frame_time = 0.0

    while cap.isOpened():
        now = time.time()
        if now - last_frame_time < frame_interval:
            cv2.waitKey(1)
            continue
        last_frame_time = now

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        results = tracker.find_hands(frame)
        landmarks = tracker.get_landmarks(results, smooth=True)
        hand_confidence = tracker.get_hand_confidence()

        status_text = "Esperando mano..."
        color = COLOR_INFO

        if landmarks and len(landmarks) == 63 and hand_confidence > 0.4:
            tracker.draw_landmarks(frame, landmarks)

            prediction_ready = False
            prediction_input = None

            if is_lstm:
                sequence_buffer.append(landmarks)
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    raw_data = np.array(sequence_buffer)
                    scaled_data = scaler.transform(raw_data)
                    prediction_input = np.expand_dims(scaled_data, axis=0)
                    prediction_ready = True
                else:
                    status_text = f"Capturando... {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
            else:
                raw_data = np.array(landmarks).reshape(1, -1)
                prediction_input = scaler.transform(raw_data)
                prediction_ready = True

            if prediction_ready:
                prediction = model.predict(prediction_input, verbose=0)
                confidence = float(np.max(prediction))
                class_index = int(np.argmax(prediction))
                detected_letter = labels[class_index]

                if confidence > AI_CONFIDENCE_THRESHOLD:
                    status_text = f"Letra: {detected_letter} ({int(confidence * 100)}%)"
                    color = COLOR_SUCCESS

                    if detected_letter != last_prediction:
                        voice.speak(detected_letter)
                        last_prediction = detected_letter
                        if is_lstm:
                            sequence_buffer.clear()

                    if game.game_active:
                        is_correct, game_msg = game.check_prediction(detected_letter)
                        status_text = game_msg

                        if is_correct:
                            voice.speak("Correcto")
                            if game.game_active:
                                voice.speak(game.target_letter)
                        elif not game.game_active:
                            # El tiempo se agotó dentro de check_prediction
                            show_summary = True
                            voice.speak(game_msg)
                else:
                    status_text = "Seña poco clara..."
                    color = COLOR_DANGER
        else:
            if is_lstm and sequence_buffer:
                sequence_buffer.clear()

        # ── Renderizado HUD ──────────────────────────────────────────
        draw_hud(frame, status_text, color, game)

        if show_summary and not game.game_active:
            draw_summary(frame, game)

        cv2.imshow(window_name, frame)

        # ── Teclado ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('k') or key == ord('K') or key == 27:
            break

        elif key == ord('j') or key == ord('J'):
            show_summary = False
            target = game.start_game()
            tracker.reset_smoothing()
            if is_lstm:
                sequence_buffer.clear()
            voice.speak(f"Comienza. Forma la letra {target}")

        elif key in DIFFICULTY_KEYS:
            diff = DIFFICULTY_KEYS[key]
            show_summary = False
            target = game.start_game(difficulty=diff)
            tracker.reset_smoothing()
            if is_lstm:
                sequence_buffer.clear()
            voice.speak(f"Dificultad {diff}. Forma la letra {target}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()