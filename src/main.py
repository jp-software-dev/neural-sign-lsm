# Punto de entrada y orquestador principal de la aplicación.
# Gestiona el bucle de video, la máquina de estados y la renderización de la UI.
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
from src.ai_engine.hand_tracking import HandTracker # type: ignore
from src.modules import VoiceAssistant, SignGame, GameMode, profile_manager, sound_manager

FONT = cv2.FONT_HERSHEY_SIMPLEX

HUD_BG_COLOR = (20, 20, 20)
STREAK_COLOR = (0, 255, 255)
WORD_COLOR = (255, 255, 255)
PROGRESS_COLOR = (180, 255, 180)

def draw_hud(frame, status_text, color, game):
    # Dibuja el HUD principal con el estado actual y la información del juego.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    bar_height = 105 if (game and game.game_active) else 45
    cv2.rectangle(overlay, (0, 0), (w, bar_height), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, status_text, (15, 30), FONT, 0.8, color, 2, cv2.LINE_AA)

    if game and game.game_active:
        progress = game.progress_display
        cv2.putText(frame, f"Palabra: {progress}", (15, 65), FONT, 0.8, WORD_COLOR, 2, cv2.LINE_AA)
        hud_right = (
            f"Pts: {game.score} | "
            f"Racha: {game.streak}x | " if game.MODE == GameMode.GAME else ""
            f"Palabras: {game.words_completed} | "
            f"Tiempo: {game.time_left}s"
        )
        cv2.putText(frame, hud_right, (15, 95), FONT, 0.55, STREAK_COLOR, 1, cv2.LINE_AA)
        # Calcula el ancho de la barra de tiempo, asegurando que no exceda el ancho de la pantalla.
        bar_w = min(w, int(w * game.time_left / game.time_limit))
        bar_color = COLOR_SUCCESS if game.time_left > 20 else COLOR_DANGER
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), bar_color, -1)
    
    if game and game.game_active:
        hint = "Presiona 'P' para pausar"
    else:
        hint = "1: Libre | 2: Juego | 3: Puntuaciones | 4: Salir"
    cv2.putText(frame, hint, (15, h - 16), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

def draw_summary(frame, game, high_scores):
    # Dibuja la pantalla de resumen al finalizar una partida.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    x_start = int(w * 0.15)
    x_end = int(w * 0.85)
    y_start = int(h * 0.15)
    y_end = int(h * 0.85)
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    lines = [
        "SESION TERMINADA",
        f"Puntuacion: {game.score} pts",
        f"Palabras:   {game.words_completed}",
    ]
    if game.MODE == GameMode.GAME:
        lines.append(f"Mejor racha: {game.peak_streak}x")

    lines.extend(["", "MEJORES PUNTAJES:"])
    if not high_scores:
        lines.append("Aun no hay puntajes")
    else:
        for i, score_entry in enumerate(high_scores):
            lines.append(f"{i+1}. {score_entry['name']} - {score_entry['score']} pts")

    lines.extend(["", "Presiona una tecla para continuar..."])

    for i, line in enumerate(lines):
        y = y_start + 60 + i * 40
        color = WORD_COLOR if i == 0 else (220, 220, 220)
        scale = 0.85 if i == 0 else 0.65
        if "MEJORES PUNTAJES" in line:
            scale = 0.7
        cv2.putText(frame, line, (x_start + 30, y), FONT, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)

def draw_high_scores_screen(frame, high_scores):
    # Dibuja la pantalla que muestra los mejores puntajes guardados.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    x_start, x_end = int(w * 0.2), int(w * 0.8)
    y_start, y_end = int(h * 0.2), int(h * 0.8)
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    lines = ["MEJORES PUNTAJES"]
    lines.extend(["", ""])

    if not high_scores:
        lines.append("Aun no hay puntajes")
    else:
        for i, score_entry in enumerate(high_scores):
            lines.append(f"{i+1}. {score_entry['name']} - {score_entry['score']} pts")

    lines.extend(["", "", "Presiona 'B' para Borrar | Otra tecla para volver"])

    for i, line in enumerate(lines):
        y = y_start + 60 + i * 45
        color = STREAK_COLOR if i == 0 else (220, 220, 220)
        scale = 1.0 if i == 0 else 0.7
        text_size = cv2.getTextSize(line, FONT, scale, 2)[0]
        cv2.putText(frame, line, (x_start + (x_end - x_start - text_size[0]) // 2, y), FONT, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)


def draw_difficulty_menu(frame):
    # Dibuja la pantalla para que el usuario seleccione la dificultad del juego.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    x_start = int(w * 0.2)
    x_end = int(w * 0.8)
    y_start = int(h * 0.2)
    y_end = int(h * 0.8)
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    lines = [
        "SELECCIONA LA DIFICULTAD", "1: Facil", "2: Medio", "3: Dificil",
        "4: Aleatorio", "", "5: Volver"
    ]
    for i, line in enumerate(lines):
        y = y_start + 60 + i * 45
        color = WORD_COLOR if i == 0 else (220, 220, 220)
        scale = 0.8 if i == 0 else 0.7
        text_size = cv2.getTextSize(line, FONT, scale, 2)[0]
        cv2.putText(frame, line, (x_start + (x_end - x_start - text_size[0]) // 2, y), FONT, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)

def draw_name_input_screen(frame, current_name):
    # Dibuja la pantalla para que el usuario ingrese su nombre tras un nuevo récord.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    x_start, x_end = int(w * 0.2), int(w * 0.8)
    y_start, y_end = int(h * 0.2), int(h * 0.8)
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    title = "NUEVO RECORD"
    prompt = "Ingresa tu nombre:"
    
    title_size = cv2.getTextSize(title, FONT, 1.2, 2)[0]
    cv2.putText(frame, title, (x_start + (x_end - x_start - title_size[0]) // 2, y_start + 70), FONT, 1.2, STREAK_COLOR, 2, cv2.LINE_AA)

    prompt_size = cv2.getTextSize(prompt, FONT, 0.8, 1)[0]
    cv2.putText(frame, prompt, (x_start + (x_end - x_start - prompt_size[0]) // 2, y_start + 140), FONT, 0.8, WORD_COLOR, 1, cv2.LINE_AA)

    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    cv2.putText(frame, f"{current_name}{cursor}", (x_start + 50, y_start + 200), FONT, 1, WORD_COLOR, 2, cv2.LINE_AA)

def draw_delete_confirmation_screen(frame):
    # Dibuja la pantalla de confirmación para borrar puntajes.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    x_start, x_end = int(w * 0.2), int(w * 0.8)
    y_start, y_end = int(h * 0.2), int(h * 0.8)
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 0, 40), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    lines = [
        "¿BORRAR PUNTAJES?",
        "Esta accion no se puede deshacer.",
        "",
        "Presiona 'S' para confirmar",
        "Presiona otra tecla para cancelar"
    ]
    for i, line in enumerate(lines):
        y = y_start + 60 + i * 45
        color = COLOR_DANGER if i == 0 else (220, 220, 220)
        scale = 1.0 if i == 0 else 0.7
        text_size = cv2.getTextSize(line, FONT, scale, 2)[0]
        cv2.putText(frame, line, (x_start + (x_end - x_start - text_size[0]) // 2, y), FONT, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)

def draw_countdown_screen(frame, text):
    # Dibuja la pantalla de cuenta atrás.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font_scale = 6.0
    thickness = 15
    text_size = cv2.getTextSize(text, FONT, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_pause_screen(frame):
    # Dibuja la pantalla de pausa.
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.4, 0, frame)

    text = "PAUSA"
    font_scale = 3.0
    thickness = 8
    text_size = cv2.getTextSize(text, FONT, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    hint = "Presiona 'P' para reanudar"
    hint_size = cv2.getTextSize(hint, FONT, 0.8, 2)[0]
    hint_x = (w - hint_size[0]) // 2
    cv2.putText(frame, hint, (hint_x, text_y + 80), FONT, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

def load_dependencies():
    # Carga los artefactos de IA (modelo, escalador, etiquetas) al iniciar.
    try:
        model = load_model(AI_MODEL_PATH)
        with open(LABELS_JSON_PATH, 'r') as f:
            labels_dict = json.load(f)
        labels = {int(k): v for k, v in labels_dict.items()}
        available_letters = list(labels.values())
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        is_lstm = len(model.input_shape) == 3
        print("[INFO] Dependencias cargadas correctamente.")
        return model, scaler, labels, available_letters, is_lstm
    except Exception as e:
        print(f"[ERROR] Crítico al cargar dependencias: {e}")
        return None, None, None, None, False

def process_prediction(landmarks, scaler, model, labels, is_lstm, sequence_buffer):
    # Toma los landmarks, los procesa y devuelve la predicción del modelo.
    prediction_input = None
    if is_lstm:
        sequence_buffer.append(landmarks)
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            raw_data = np.array(sequence_buffer)
            scaled_data = scaler.transform(raw_data)
            prediction_input = np.expand_dims(scaled_data, axis=0)
    else:
        raw_data = np.array(landmarks).reshape(1, -1)
        prediction_input = scaler.transform(raw_data)

    if prediction_input is not None:
        prediction = model.predict(prediction_input, verbose=0)
        confidence = float(np.max(prediction))
        class_index = int(np.argmax(prediction))
        detected_letter = labels.get(class_index, "?")
        return detected_letter, confidence
    
    return None, 0.0

def handle_stability(detected_letter, confidence, candidate_letter, stable_frames):
    # Asegura que una seña se mantenga estable por N frames antes de confirmarla.
    confirmed_letter = None
    status_text = ""
    color = COLOR_DANGER

    if confidence > AI_CONFIDENCE_THRESHOLD:
        if detected_letter == candidate_letter:
            stable_frames += 1
        else:
            candidate_letter = detected_letter
            stable_frames = 1
        
        REQUIRED_FRAMES = 5
        if stable_frames >= REQUIRED_FRAMES:
            confirmed_letter = candidate_letter
            status_text = f"Letra: {confirmed_letter} ({int(confidence * 100)}%)"
            color = COLOR_SUCCESS
        else:
            status_text = f"Sosten la seña... {stable_frames}/{REQUIRED_FRAMES}"
            color = (0, 165, 255)
    else:
        candidate_letter = ""
        stable_frames = 0
        status_text = "Sena poco clara..."
        color = COLOR_DANGER

    return candidate_letter, stable_frames, confirmed_letter, status_text, color

def reset_prediction_state(tracker, is_lstm, sequence_buffer):
    # Limpia el estado de predicción al cambiar de modo o juego.
    tracker.reset_smoothing()
    if is_lstm and sequence_buffer is not None:
        sequence_buffer.clear()
    return "", 0, ""

def main():
    # Inicializa y ejecuta el bucle principal de la aplicación.
    model, scaler, labels, available_letters, is_lstm = load_dependencies()
    if not model:
        return

    sound_manager.init()
    tracker = HandTracker(ema_alpha=0.3)
    voice = VoiceAssistant()
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) if is_lstm else None
    
    game = None
    cap = cv2.VideoCapture(0)
    window_name = "NeuralSign-LSM - Traductor de Lengua de Senas Mexicana"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    app_state = "menu" # "menu", "selecting_difficulty", "countdown", "in_game", "paused", "entering_name", "summary", "viewing_scores", "confirm_delete"
    countdown_start_time = 0.0
    countdown_stage = 0
    game_class_to_start = None
    last_prediction = ""
    stable_frames = 0
    candidate_letter = ""
    high_scores = profile_manager.load_high_scores()
    current_name = ""
    final_score = 0
    status_text = "Iniciando..."
    color = COLOR_INFO

    voice.speak("Sistema iniciado. Presiona 1 para Modo Libre, 2 para Modo Juego, 3 para ver Puntuaciones, o 4 para Salir.")
    time.sleep(2)

    frame_interval = 1.0 / FPS_LIMIT

    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        elapsed_time = time.time() - start_time
        wait_time = max(1, int((frame_interval - elapsed_time) * 1000))
        key = cv2.waitKey(wait_time) & 0xFF

        if app_state == "selecting_difficulty":
            draw_difficulty_menu(frame)
            if key != 255:
                difficulty = None
                if key == ord('1'): difficulty = "facil"
                elif key == ord('2'): difficulty = "medio"
                elif key == ord('3'): difficulty = "dificil"
                elif key == ord('4'): difficulty = "aleatorio"
                elif key == ord('5') or key == 27:
                    app_state = "menu"
                    game_class_to_start = None
                    voice.speak("Seleccion cancelada.")

                if difficulty and game_class_to_start:
                    game = game_class_to_start(available_letters, difficulty=difficulty)
                    app_state = "countdown"
                    countdown_stage = 0
                    countdown_start_time = time.time()
                    voice.speak("Prepárate")

        elif app_state == "countdown":
            elapsed = time.time() - countdown_start_time
            countdown_text = ""

            if elapsed < 1:
                countdown_text = "3"
                if countdown_stage == 0:
                    sound_manager.play("tick")
                    countdown_stage = 1
            elif elapsed < 2:
                countdown_text = "2"
                if countdown_stage == 1:
                    sound_manager.play("tick")
                    countdown_stage = 2
            elif elapsed < 3:
                countdown_text = "1"
                if countdown_stage == 2:
                    sound_manager.play("tick")
                    countdown_stage = 3
            elif elapsed < 3.7:
                countdown_text = "YA!"
                if countdown_stage == 3:
                    sound_manager.play("finish")
                    countdown_stage = 4
            else:
                app_state = "in_game"
                target = game.start_game()
                last_prediction, stable_frames, candidate_letter = reset_prediction_state(tracker, is_lstm, sequence_buffer)
                if game.MODE == GameMode.GAME:
                    voice.speak(f"Tu primera letra es {target}")
            
            if app_state == "countdown":
                draw_countdown_screen(frame, countdown_text)

        elif app_state == "entering_name":
            draw_name_input_screen(frame, current_name)
            if key != 255:
                if key == 8:
                    current_name = current_name[:-1]
                elif key == 13:
                    if current_name:
                        high_scores = profile_manager.add_high_score(current_name, final_score, high_scores)
                        profile_manager.save_high_scores(high_scores)
                        voice.speak(f"Puntaje de {current_name} guardado.")
                        app_state = "summary"
                elif 32 <= key <= 126 and len(current_name) < 10:
                    current_name += chr(key).upper()

        elif app_state == "summary":
            draw_summary(frame, game, high_scores)
            if key != 255:
                app_state = "menu"
                game = None
                last_prediction, stable_frames, candidate_letter = reset_prediction_state(tracker, is_lstm, sequence_buffer)
                voice.speak("Menu principal.")

        elif app_state == "viewing_scores":
            draw_high_scores_screen(frame, high_scores)
            if key == ord('b'):
                app_state = "confirm_delete"
                voice.speak("Confirmar borrado de puntajes.")
            elif key != 255 and key != ord('b'):
                app_state = "menu"
                voice.speak("Menu principal.")

        elif app_state == "confirm_delete":
            draw_delete_confirmation_screen(frame)
            if key == ord('s'):
                profile_manager.delete_high_scores()
                high_scores = profile_manager.load_high_scores()
                app_state = "menu"
                voice.speak("Puntajes borrados.")
            elif key != 255 and key != ord('s'):
                app_state = "viewing_scores"

        elif app_state == "paused":
            draw_hud(frame, "PAUSA", COLOR_INFO, game)
            draw_pause_screen(frame)
            if key == ord('p'):
                app_state = "in_game"
                if game: game.resume()
                voice.speak("Juego reanudado.")

        elif app_state in ["menu", "in_game"]:
            if app_state != "summary":
                landmarks = tracker.get_landmarks(tracker.find_hands(frame), smooth=True)
                hand_confidence = tracker.get_hand_confidence()

                if landmarks and hand_confidence > 0.4:
                    tracker.draw_landmarks(frame, landmarks)
                    detected_letter, confidence = process_prediction(landmarks, scaler, model, labels, is_lstm, sequence_buffer)

                    if detected_letter:
                        candidate_letter, stable_frames, confirmed, text, c = handle_stability(detected_letter, confidence, candidate_letter, stable_frames)
                        status_text, color = text, c

                        if confirmed and confirmed != last_prediction:
                            last_prediction = confirmed
                            if app_state == "in_game" and game and game.game_active:
                                is_correct, game_msg = game.check_prediction(confirmed)
                                status_text = game_msg
                                if is_correct:
                                    sound_manager.play("correct")
                                    if "¡Palabra!" in game_msg:
                                        voice.speak("Palabra completa")
                                        if game.game_active:
                                            voice.speak(f"Siguiente letra: {game.target_letter}")
                                    else:
                                        voice.speak("Correcto")
                                else:
                                    sound_manager.play("incorrect")

                                if not game.game_active:
                                    final_score = game.score
                                    if profile_manager.is_high_score(final_score, high_scores):
                                        app_state = "entering_name"
                                        current_name = ""
                                        voice.speak("Nuevo record. Ingresa tu nombre y presiona enter.")
                                    else:
                                        app_state = "summary"
                                    voice.speak(game_msg)
                            else: # Modo Libre
                                voice.speak(confirmed)
                            if is_lstm: sequence_buffer.clear()
                    elif is_lstm:
                        status_text, color = f"Capturando... {len(sequence_buffer)}/{SEQUENCE_LENGTH}", COLOR_INFO
                else:
                    status_text, color, stable_frames, candidate_letter = "Esperando mano...", COLOR_INFO, 0, ""
                    if is_lstm and sequence_buffer: sequence_buffer.clear()

            if app_state == "in_game" and game:
                expired, msg = game.check_time()
                if expired:
                    status_text = msg
                    final_score = game.score
                    if profile_manager.is_high_score(final_score, high_scores):
                        app_state = "entering_name"
                        current_name = ""
                        voice.speak("Nuevo record. Ingresa tu nombre y presiona enter.")
                    else:
                        app_state = "summary"

                    voice.speak(msg)

            draw_hud(frame, status_text, color, game)

            if key != 255:
                if key == ord('p') and app_state == "in_game" and game:
                    app_state = "paused"
                    game.pause()
                    voice.speak("Juego pausado.")
                elif key == ord('4') or key == 27:
                    break
                elif key == ord('1'):
                    app_state = "menu"
                    game = None
                    last_prediction, stable_frames, candidate_letter = reset_prediction_state(tracker, is_lstm, sequence_buffer)
                    voice.speak("Traduccion libre activada.")
                elif key == ord('2'):
                    app_state = "selecting_difficulty"
                    game_class_to_start = SignGame
                    game = None
                    voice.speak("Selecciona la dificultad para el modo de juego.")
                elif key == ord('3'):
                    app_state = "viewing_scores"
                    voice.speak("Mostrando mejores puntajes.")

        cv2.imshow(window_name, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()