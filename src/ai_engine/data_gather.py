import cv2
import csv
import os
import time
import numpy as np
from src.config.settings import CAMERA_WIDTH, CAMERA_HEIGHT, LANDMARKS_CSV_PATH
from src.utils import app_logger
from src.ai_engine.hand_tracking import HandTracker

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Folder created: {directory}")

def save_data(label, landmarks):
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
    seq_dir = os.path.join("data", "sequences", label)
    os.makedirs(seq_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = os.path.join(seq_dir, f"{timestamp}.npy")
    np.save(filename, np.array(sequence))
    print(f"[OK] Sequence saved to {filename}")

def record_sequence(tracker, cap, label, seq_len=30):
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    tracker = HandTracker(ema_alpha=0.25)
    last_detected_label = None

    print("Camera ready. Instructions:")
    print("  - Press a LOWERCASE letter (a-z) to set the current letter (e.g., 'a').")
    print("  - Then use UPPERCASE letters for actions:")
    print("      [C] Capture 1 static sample")
    print("      [V] Capture 500 fast static samples")
    print("      [B] Record a movement sequence (30 frames)")
    print("      [Q] Quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            frame = cv2.flip(frame, 1)
            tracker.process_frame(frame)
            frame = tracker.draw_hands(frame)
            landmarks = tracker.get_landmarks(smooth=True)

            if last_detected_label:
                cv2.putText(frame, f"Current letter: {last_detected_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Press a-z to choose a letter", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, "[C] 1 sample | [V] 500 samples | [B] sequence | [Q] quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("NeuralSign-LSM : Data Gathering", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

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