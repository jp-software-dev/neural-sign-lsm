import cv2
import numpy as np
import json
import sys
import os
import pickle
import time
from collections import deque

# Inyectamos dinámicamente la ruta absoluta del directorio raíz en el entorno de ejecución para garantizar la resolución determinista de los módulos arquitectónicos internos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from src.config.settings import (
    AI_MODEL_PATH, LABELS_JSON_PATH, SCALER_PATH,
    SEQUENCE_LENGTH, AI_CONFIDENCE_THRESHOLD,
    COLOR_SUCCESS, COLOR_DANGER, COLOR_INFO, FPS_LIMIT,
)
from src.ai_engine.hand_tracking import HandTracker
from src.modules import VoiceAssistant, SpeedGame, SpellingGame, GameMode


# ── Constantes de UI ────────────────────────────────────────────────────

# Definimos el espacio de variables estáticas para la parametrización tipográfica y los vectores de color matricial en formato BGR de la interfaz gráfica
FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_BG_COLOR = (10, 10, 10)
STREAK_COLOR = (0, 215, 255)   
WORD_COLOR = (255, 255, 80)    
PROGRESS_COLOR = (180, 255, 180)

# Mapeamos los identificadores ASCII del teclado físico a descriptores léxicos para el enrutamiento de la entropía de los motores de juego
DIFFICULTY_KEYS = {
    ord('1'): "facil",
    ord('2'): "medio",
    ord('3'): "dificil",
}
GAME2_KEYS = {
    ord('4'): "facil",
    ord('5'): "medio",
    ord('6'): "dificil",
}


# ── Helpers de dibujo ───────────────────────────────────────────────────

def draw_hud(frame, status_text, color, game):
    # Proyectamos la capa de telemetría (HUD) superpuesta al flujo de video nativo mediante operaciones de alpha blending
    h, w = frame.shape[:2]

    # Construimos un plano oscuro semitransparente para maximizar el contraste de las primitivas de texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Inyectamos el vector de estado principal del clasificador probabilístico
    cv2.putText(frame, status_text, (15, 35), FONT, 0.8, color, 2, cv2.LINE_AA)

    if game and game.game_active:
        # Renderizamos dinámicamente el buffer de progreso iterativo de la palabra objetivo
        progress = game.progress_display
        cv2.putText(frame, f"Palabra: {progress}", (15, 68), FONT, 0.8, WORD_COLOR, 2, cv2.LINE_AA)

        # Concatenamos los escalares de rendimiento (KPIs) de la sesión en un plano secuencial
        hud_right = (
            f"Pts:{game.score} | "
            f"Racha:{game.streak}x | " if game.MODE == GameMode.SPEED else ""
            f"Palabras:{game.words_completed} | "
            f"Tiempo:{game.time_left}s"
        )
        cv2.putText(frame, hud_right, (15, 100), FONT, 0.55, STREAK_COLOR, 1, cv2.LINE_AA)

        # Dibujamos un indicador analógico de degradación temporal condicionado por el estado crítico del límite
        bar_w = int(w * game.time_left / game.time_limit)
        bar_color = COLOR_SUCCESS if game.time_left > 20 else COLOR_DANGER
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), bar_color, -1)

    # Desplegamos la leyenda estática de manipulación I/O
    hint = "1-3:Veloz | 4-6:Deletreo | K:Salir"
    cv2.putText(frame, hint, (15, h - 16), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_summary(frame, game):
    # Generamos una máscara de enfoque focal (overlay) para reportar la analítica resultante de la destrucción del ciclo de evaluación
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 4, h // 4), (w * 3 // 4, h * 3 // 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    lines = [
        "SESION TERMINADA",
        f"Puntuacion: {game.score} pts",
        f"Palabras:   {game.words_completed}",
    ]
    # Aplicamos polimorfismo visual según el modelo de datos subyacente para desplegar el multiplicador pico o la tasa de fallos
    if game.MODE == GameMode.SPEED:
        lines.append(f"Mejor racha: {game.peak_streak}x")
    elif game.MODE == GameMode.SPELLING:
        lines.append(f"Errores:    {game.total_errors}")

    lines.extend(["", "1-6: Nueva sesion | K: Salir"])

    for i, line in enumerate(lines):
        y = h // 4 + 40 + i * 34
        color = WORD_COLOR if i == 0 else (220, 220, 220)
        scale = 0.85 if i == 0 else 0.7
        cv2.putText(frame, line, (w // 4 + 20, y), FONT, scale, color, 2 if i == 0 else 1, cv2.LINE_AA)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    # Instanciamos el subsistema de visión computacional inyectando un filtro de media móvil exponencial (EMA) para suavizado de contornos
    tracker = HandTracker(ema_alpha=0.25)
    # Inicializamos el orquestador asíncrono de síntesis acústica
    voice = VoiceAssistant()

    # Ejecutamos el pipeline de recuperación de los artefactos serializados resultantes del entorno de entrenamiento
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

    # Evaluamos dinámicamente la topología del tensor de entrada de la red para inferir la presencia de una capa recurrente
    is_lstm = len(model.input_shape) == 3
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) if is_lstm else None
    print(f"Modo de inferencia: {'LSTM' if is_lstm else 'Estático'}")

    game = None

    # Asignamos el descriptor lógico del hardware de captura óptica y parametrizamos el lienzo de proyección nativa
    cap = cv2.VideoCapture(0)
    window_name = "NeuralSign-LSM - Traductor de Lengua de Señas Mexicana"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    last_prediction = ""
    show_summary = False
    voice.speak("Sistema iniciado. Presiona 1 a 3 para juego de velocidad, 4 a 6 para deletreo, o K para salir.")

    # Implementamos un gobernador de reloj (throttle) estricto basado en deltas para evitar sobrecarga del bus de la ALU y la GPU
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

        # Invertimos el eje horizontal de la matriz RGB para mimetizar la perspectiva especular de la anatomía
        frame = cv2.flip(frame, 1)
        # Sometemos el plano focal a la extracción de características del modelo de topología subyacente
        results = tracker.find_hands(frame)

        # Forzamos la evaluación temporal de la sesión para evitar bloqueos del sistema de recompensas en caso de oclusión anatómica
        if game and game.game_active:
            expired, msg = game.check_time()
            if expired:
                status_text = msg
                show_summary = True
                
                # Consumimos la cola pendiente de eventos sonoros de forma prioritaria ante la muerte del proceso
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

        # Validamos la estructura dimensional del vector morfológico y aplicamos una compuerta lógica basada en el umbral probabilístico espacial
        if landmarks and len(landmarks) == 63 and hand_confidence > 0.4:
            tracker.draw_landmarks(frame, landmarks)

            prediction_ready = False
            prediction_input = None

            if is_lstm:
                # Acumulamos el vector en la ventana deslizante hasta completar la secuencia requerida por la memoria a corto plazo
                sequence_buffer.append(landmarks)
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    raw_data = np.array(sequence_buffer)
                    scaled_data = scaler.transform(raw_data)
                    # Expandimos la dimensión escalar para acomodar la firma estructural del tensor predictivo del modelo
                    prediction_input = np.expand_dims(scaled_data, axis=0)
                    prediction_ready = True
                else:
                    status_text = f"Capturando... {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
            else:
                # Proyectamos las coordenadas estáticas sobre el mismo espacio de normalización estadística del conjunto de entrenamiento
                raw_data = np.array(landmarks).reshape(1, -1)
                prediction_input = scaler.transform(raw_data)
                prediction_ready = True

            if prediction_ready:
                # Ejecutamos un pase hacia adelante (forward pass) sobre el grafo computacional silenciando el flujo I/O de la consola
                prediction = model.predict(prediction_input, verbose=0)
                confidence = float(np.max(prediction))
                class_index = int(np.argmax(prediction))
                detected_letter = labels[class_index]

                # Condicionamos la aceptación de la predicción iterativa a la convergencia sobre el límite mínimo de precisión establecido
                if confidence > AI_CONFIDENCE_THRESHOLD:
                    status_text = f"Letra: {detected_letter} ({int(confidence * 100)}%)"
                    color = COLOR_SUCCESS

                    # Interceptamos la propagación léxica para impedir colisiones acústicas con el enrutador del motor de juego
                    if detected_letter != last_prediction:
                        if not (game and game.game_active):
                            voice.speak(detected_letter)
                        last_prediction = detected_letter
                        if is_lstm:
                            # Purgamos la memoria de transición para evitar contaminación cruzada entre inferencias secuenciales
                            sequence_buffer.clear()

                    if game and game.game_active:
                        # Delegamos la verificación probabilística al árbol de decisiones del objeto gamificado
                        is_correct, game_msg = game.check_prediction(detected_letter)
                        status_text = game_msg

                        # Consumimos de forma polimórfica los descriptores fonéticos residuales de ambas máquinas de estado
                        if hasattr(game, 'pending_voice') and game.pending_voice:
                            for v_msg in game.pending_voice:
                                voice.speak(v_msg)
                            game.pending_voice.clear()
                        elif is_correct: 
                            voice.speak("Correcto")
                            if game.game_active: 
                                voice.speak(game.target_letter)

                        # Detectamos señales de terminación emitidas por el núcleo de validación léxica para desplegar la telemetría histórica
                        if not game.game_active and not show_summary:
                            show_summary = True
                            if not hasattr(game, 'pending_voice'):
                                voice.speak(game_msg)

                else:
                    status_text = "Seña poco clara..."
                    color = COLOR_DANGER
        else:
            if is_lstm and sequence_buffer:
                sequence_buffer.clear()

        # ── Renderizado HUD ──────────────────────────────────────────
        draw_hud(frame, status_text, color, game)

        if show_summary and game and not game.game_active:
            draw_summary(frame, game)

        cv2.imshow(window_name, frame)

        # ── Interceptación de Interrupciones de Teclado ───────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('k') or key == ord('K') or key == 27:
            # Capturamos la señal de escape para invocar la subrutina de destrucción controlada
            break

        elif key in DIFFICULTY_KEYS:
            diff = DIFFICULTY_KEYS[key]
            show_summary = False
            # Instanciamos un nuevo pipeline de velocidad y refrescamos el modelo analítico reseteando los acumuladores
            game = SpeedGame(available_letters, difficulty=diff)
            target = game.start_game()
            tracker.reset_smoothing()
            if is_lstm:
                sequence_buffer.clear()
            voice.speak(f"Juego de velocidad, dificultad {diff}. Forma la letra {target}")

        elif key in GAME2_KEYS:
            diff = GAME2_KEYS[key]
            show_summary = False
            # Construimos la topología del juego de dictado y purgamos el estado transicional anterior
            game = SpellingGame(available_letters, difficulty=diff)
            _ = game.start_game() 
            tracker.reset_smoothing()
            if is_lstm:
                sequence_buffer.clear()

            # Despachamos las instrucciones de inicialización SAPI al hilo asíncrono secundario
            for msg in game.pending_voice:
                voice.speak(msg)
            game.pending_voice.clear()

    # Liberamos los descriptores de hardware devolviendo el control al kernel del sistema operativo
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()