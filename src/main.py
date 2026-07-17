import cv2
import numpy as np
import json
import sys
import os
import pickle
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model # type: ignore
from src.config.settings import (
    AI_MODEL_PATH, LABELS_JSON_PATH, SCALER_PATH,
    SEQUENCE_LENGTH, AI_CONFIDENCE_THRESHOLD,
    COLOR_SUCCESS, COLOR_DANGER, COLOR_INFO, FPS_LIMIT,
)
from src.ai_engine.hand_tracking import HandTracker
from src.modules import VoiceAssistant, SpeedGame, SpellingGame, GameMode

FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_BG_COLOR = (10, 10, 10)
STREAK_COLOR = (0, 215, 255)   
WORD_COLOR = (255, 255, 80)    
PROGRESS_COLOR = (180, 255, 180)

def draw_hud(frame, status_text, color, game):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    bar_height = 105 if (game and game.game_active) else 45
    cv2.rectangle(overlay, (0, 0), (w, bar_height), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    
    cv2.putText(frame, status_text, (15, 30), FONT, 0.8, color, 2, cv2.LINE_AA)

    if game and game.game_active:
        progress = game.progress_display
        cv2.putText(frame, f"Palabra: {progress}", (15, 65), FONT, 0.8, WORD_COLOR, 2, cv2.LINE_AA)
        hud_right = (
            f"Pts: {game.score} | "
            f"Racha: {game.streak}x | " if game.MODE == GameMode.SPEED else ""
            f"Palabras: {game.words_completed} | "
            f"Tiempo: {game.time_left}s"
        )
        cv2.putText(frame, hud_right, (15, 95), FONT, 0.55, STREAK_COLOR, 1, cv2.LINE_AA)
        bar_w = int(w * game.time_left / game.time_limit)
        bar_color = COLOR_SUCCESS if game.time_left > 20 else COLOR_DANGER
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), bar_color, -1)

    hint = "1: Libre | 2: Velocidad | 3: Deletreo | 4: Salir"
    cv2.putText(frame, hint, (15, h - 16), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

def draw_summary(frame, game):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    x_start = int(w * 0.15)
    x_end = int(w * 0.85)
    y_start = int(h * 0.20)
    y_end = int(h * 0.82)
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    lines = [
        "SESION TERMINADA",
        f"Puntuacion: {game.score} pts",
        f"Palabras:   {game.words_completed}",
    ]
    if game.MODE == GameMode.SPEED:
        lines.append(f"Mejor racha: {game.peak_streak}x")
    elif game.MODE == GameMode.SPELLING:
        lines.append(f"Errores:    {game.total_errors}")

    lines.extend(["", "1: Libre | 2: Velocidad | 3: Deletreo", "                 4: Salir"])

    for i, line in enumerate(lines):
        y = y_start + 60 + i * 40
        color = WORD_COLOR if i == 0 else (220, 220, 220)
        scale = 0.85 if i == 0 else 0.7
        cv2.putText(frame, line, (x_start + 30, y), FONT, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)

def main():
    # Reducimos el suavizado (alpha) para que la lectura de la mano sea más ágil y responsiva
    tracker = HandTracker(ema_alpha=0.5)
    voice = VoiceAssistant()

    try:
        model = load_model(AI_MODEL_PATH)
        with open(LABELS_JSON_PATH, 'r') as f:
            labels_dict = json.load(f)
        labels = {int(k): v for k, v in labels_dict.items()}
        available_letters = list(labels.values())
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    is_lstm = len(model.input_shape) == 3
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) if is_lstm else None

    game = None
    cap = cv2.VideoCapture(0)
    window_name = "NeuralSign-LSM - Traductor de Lengua de Senas Mexicana"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    last_prediction = ""
    show_summary = False
    
    # --- VARIABLES DE RIGUROSIDAD ---
    STRICT_THRESHOLD = 0.85      # Exige 85% de certeza absoluta
    REQUIRED_FRAMES = 5          # La mano debe mantenerse congelada y correcta por 5 frames seguidos
    stable_frames = 0
    candidate_letter = ""
    # --------------------------------

    voice.speak("Sistema iniciado. Presiona 1 para traduccion libre, 2 para velocidad, 3 para deletreo, o 4 para salir.")
    time.sleep(2) # Pausa para dar tiempo al motor de voz a inicializar

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

        if game and game.game_active:
            expired, msg = game.check_time()
            if expired:
                status_text = msg
                show_summary = True
                if hasattr(game, 'pending_voice') and game.pending_voice:
                    for v_msg in game.pending_voice:
                        voice.speak(v_msg)
                    game.pending_voice.clear()
                else: 
                    voice.speak(msg)

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
                detected_letter = labels.get(class_index, "?")

                # CANDADO 1: Filtro matemático de alta precisión
                if confidence > STRICT_THRESHOLD:
                    
                    # CANDADO 2: Acumulador de estabilidad temporal
                    if detected_letter == candidate_letter:
                        stable_frames += 1
                    else:
                        candidate_letter = detected_letter
                        stable_frames = 1

                    # Si el usuario sostuvo la seña clara y firme durante los frames requeridos
                    if stable_frames >= REQUIRED_FRAMES:
                        status_text = f"Letra: {detected_letter} ({int(confidence * 100)}%)"
                        color = COLOR_SUCCESS

                        if detected_letter != last_prediction:
                            if not (game and game.game_active):
                                voice.speak(detected_letter)
                            last_prediction = detected_letter
                            if is_lstm:
                                sequence_buffer.clear()

                        if game and game.game_active:
                            is_correct, game_msg = game.check_prediction(detected_letter)
                            status_text = game_msg

                            if hasattr(game, 'pending_voice') and game.pending_voice:
                                for v_msg in game.pending_voice:
                                    voice.speak(v_msg)
                                game.pending_voice.clear()
                            elif is_correct: 
                                voice.speak("Correcto")
                                if game.game_active: 
                                    voice.speak(game.target_letter)

                            if not game.game_active and not show_summary:
                                show_summary = True
                                if not hasattr(game, 'pending_voice'):
                                    voice.speak(game_msg)
                    else:
                        # Feedback en pantalla de que la está haciendo bien pero necesita sostenerla
                        status_text = f"Sosten la seña... {stable_frames}/{REQUIRED_FRAMES}"
                        color = (0, 165, 255) # Naranja
                else:
                    stable_frames = 0 # Resetea la cuenta si detecta una anomalía
                    status_text = "Sena poco clara..."
                    color = COLOR_DANGER
        else:
            stable_frames = 0
            if is_lstm and sequence_buffer:
                sequence_buffer.clear()

        draw_hud(frame, status_text, color, game)

        if show_summary and game and not game.game_active:
            draw_summary(frame, game)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('4') or key == 27:
            break
        elif key == ord('1'):
            show_summary = False
            game = None
            tracker.reset_smoothing()
            stable_frames = 0
            if is_lstm:
                sequence_buffer.clear()
            voice.speak("Traduccion libre activada.")
        elif key == ord('2'):
            show_summary = False
            game = SpeedGame(available_letters, difficulty="aleatorio")
            target = game.start_game()
            tracker.reset_smoothing()
            stable_frames = 0
            if is_lstm:
                sequence_buffer.clear()
            voice.speak(f"Evaluacion de velocidad. Tu primera letra es {target}")
        elif key == ord('3'):
            show_summary = False
            game = SpellingGame(available_letters, difficulty="aleatorio")
            _ = game.start_game() 
            tracker.reset_smoothing()
            stable_frames = 0
            if is_lstm:
                sequence_buffer.clear()
            voice.speak("Evaluacion de deletreo iniciada.")
            for msg in game.pending_voice:
                voice.speak(msg)
            game.pending_voice.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()