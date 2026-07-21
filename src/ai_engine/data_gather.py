# Script para la recolección de datos de señas.
# Permite capturar landmarks de manos para señas estáticas y secuencias de movimiento,
# guardándolos en formato CSV o NPY para el posterior entrenamiento del modelo.
import cv2
import csv
import os
import time
import numpy as np
from src.config.settings import CAMERA_WIDTH, CAMERA_HEIGHT, LANDMARKS_CSV_PATH
from src.utils import app_logger
from src.ai_engine.hand_tracking import HandTracker

def ensure_dir(file_path):
    # Asegura que el directorio para un archivo dado exista.
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Folder created: {directory}")

def show_reference_image(frame, label, size=180):
    # Carga y superpone una imagen de referencia de la seña en la esquina superior derecha.
    # Asume que las imágenes están en 'data/reference_images/' y se llaman 'A.png', 'B.png', etc.
    reference_dir = os.path.join("data", "reference_images")
    img_path = os.path.join(reference_dir, f"{label}.png")

    if not os.path.exists(img_path):
        cv2.putText(frame, "No ref.", (frame.shape[1] - 85, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return

    ref_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        return

    # Redimensionar manteniendo la proporción
    h, w = ref_img.shape[:2]
    scale = size / h
    new_w, new_h = int(w * scale), size
    ref_img = cv2.resize(ref_img, (new_w, new_h))

    # Posición en la esquina superior derecha
    x_offset = frame.shape[1] - new_w - 10
    y_offset = 10

    # Superponer con canal alfa (transparencia)
    if ref_img.shape[2] == 4:
        alpha_s = ref_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                (alpha_s * ref_img[:, :, c] + alpha_l * frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c])
    else: # Si no hay canal alfa
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = ref_img

def save_data(label, landmarks):
    # Guarda una única muestra de landmarks (fila) en el archivo CSV.
    try:
        ensure_dir(LANDMARKS_CSV_PATH)
        if not landmarks or len(landmarks) != 63:
            print(f"[ERROR] Invalid landmarks: {len(landmarks)} values (expected 63)")
            return False
        row = [label] + landmarks
        with open(LANDMARKS_CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[OK] Saved {label}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not save {label}: {e}")
        return False

def auto_capture(tracker, cap, label, num_samples=500, delay=0.0):
    # Captura automáticamente un número definido de muestras estáticas.
    captured = 0
    for i in range(num_samples):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame during auto capture")
            break
        frame = cv2.flip(frame, 1)
        tracker.process_frame(frame)
        landmarks = tracker.get_landmarks(smooth=True)
        if landmarks and len(landmarks) == 63:
            if save_data(label, landmarks):
                captured += 1
        frame_show = tracker.draw_hands(frame)
        cv2.putText(frame_show, f"Auto: {label} - {captured}/{num_samples}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("NeuralSign-LSM : Data Gathering", frame_show)
        cv2.waitKey(1)
        if delay > 0:
            time.sleep(delay)
    print(f"Auto capture finished. Saved {captured} samples for {label}.\n")

def save_sequence(label, sequence):
    # Guarda una secuencia de landmarks en un archivo .npy.
    seq_dir = os.path.join("data", "sequences", label)
    os.makedirs(seq_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = os.path.join(seq_dir, f"{timestamp}.npy")
    np.save(filename, np.array(sequence))
    print(f"[OK] Sequence saved to {filename}")

def record_sequence(tracker, cap, label, seq_len=30):
    # Graba una secuencia de movimiento de una longitud definida.
    print(f"Recording sequence for {label}")
    buffer = []
    print("Preparing... 3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("Recording")
    for i in range(seq_len):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            return
        frame = cv2.flip(frame, 1)
        tracker.process_frame(frame)
        landmarks = tracker.get_landmarks(smooth=True)
        if len(landmarks) != 63:
            print(f"Frame {i+1}: hand not detected, recording cancelled")
            return
        buffer.append(landmarks)
        frame_show = tracker.draw_hands(frame)
        cv2.putText(frame_show, f"Recording {label} - {i+1}/{seq_len}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("NeuralSign-LSM : Data Gathering", frame_show)
        cv2.waitKey(1)
        time.sleep(0.05)
    if len(buffer) == seq_len:
        save_sequence(label, buffer)
        print(f"Sequence for {label} saved successfully")
    else:
        print("Incomplete sequence, not saved")

def main():
    # Función principal para ejecutar el script de recolección de datos.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    tracker = HandTracker(ema_alpha=0.25)
    last_detected_label = None

    # --- Instrucciones de Uso ---
    print("Camera ready. Instructions:")
    print("  - Press a LOWERCASE letter (a-z) to set the current letter (e.g., 'a').")
    print("  - Then use UPPERCASE letters for actions:")
    print("      [C] Capture 1 static sample")
    print("      [V] Capture 500 fast static samples")
    print("      [B] Record a movement sequence (30 frames)")
    print("      [Q] Quit")

    try:
        # --- Bucle Principal de Captura ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            frame = cv2.flip(frame, 1)
            tracker.process_frame(frame)
            frame = tracker.draw_hands(frame)
            landmarks = tracker.get_landmarks(smooth=True)

            # --- Renderizado de Información en Pantalla (HUD) ---
            if last_detected_label:
                show_reference_image(frame, last_detected_label)
                cv2.putText(frame, f"Current letter: {last_detected_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Press a-z to choose a letter", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, "[C] 1 sample | [V] 500 samples | [B] sequence | [Q] quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("NeuralSign-LSM : Data Gathering", frame)

            # --- Manejo de Teclas ---
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            # Acciones basadas en la tecla presionada
            if key == ord('Q') or key == 27:
                break
            elif key == ord('C'):
                if last_detected_label and landmarks and len(landmarks) == 63:
                    save_data(last_detected_label, landmarks)
                else:
                    print("No letter defined or hand not detected")
            elif key == ord('V'):
                if last_detected_label and landmarks and len(landmarks) == 63:
                    auto_capture(tracker, cap, last_detected_label, num_samples=500, delay=0.0)
                else:
                    print("No letter defined or hand not detected")
            elif key == ord('B'):
                if last_detected_label:
                    record_sequence(tracker, cap, last_detected_label, seq_len=30)
                else:
                    print("First define a letter by pressing a-z")
            elif (97 <= key <= 122) or (65 <= key <= 90):
                last_detected_label = chr(key).upper()
                print(f"[*] Ready to record letter: {last_detected_label}")
            else:
                pass

    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":
    main()