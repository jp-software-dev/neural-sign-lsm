import cv2
import csv
import os
import time
import numpy as np
from src.config.settings import CAMERA_WIDTH, CAMERA_HEIGHT, LANDMARKS_CSV_PATH
from src.utils import app_logger
from src.ai_engine.hand_tracking import HandTracker

def ensure_dir(file_path):
    # Desensamblamos el argumento extrayendo unicamente los descriptores de ruta omitiendo la extension e indice final del archivo
    directory = os.path.dirname(file_path)
    # Ejecutamos una verificacion escalar a nivel del SO disparando subrutinas para generar la cascada de directorios si es requerida
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Folder created: {directory}")

def save_data(label, landmarks):
    try:
        # Purgamos fallos de I/O forzando una pre-verificacion sobre la persistencia jerarquica del arbol de rutas designado
        ensure_dir(LANDMARKS_CSV_PATH)
        # Interceptamos inserciones corrompidas midiendo que la topologia mantenga estricta uniformidad vectorial de sesenta y tres posiciones
        if not landmarks or len(landmarks) != 63:
            print(f"[ERROR] Invalid landmarks: {len(landmarks)} values (expected 63)")
            return False
        # Insertamos dinamicamente el identificador caracterizado en la posicion index cero para alinear el batch al esquema tabular
        row = [label] + landmarks
        # Activamos el constructor del stream abriendo comunicacion exclusiva con el descriptor I/O en formato append no obstructivo
        with open(LANDMARKS_CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Inyectamos serialmente el bloque contiguo transformando las celdas logicas en caracteres ASCII limitados
            writer.writerow(row)
        print(f"[OK] Saved {label}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not save {label}: {e}")
        return False

def auto_capture(tracker, cap, label, num_samples=500, delay=0.0):
    # Instanciamos el iterador logico limitando superiormente el techo transaccional basandose en la directiva inyectada de parametro
    captured = 0
    for i in range(num_samples):
        # Desempaquetamos la devolucion de llamada hardware recuperando booleano de integridad y la matriz RAW consecutiva
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame during auto capture")
            break
        # Volteamos el orden columnar transformando los pixeles a modo de espejo para facilitar la orientacion visual e interaccion humana
        frame = cv2.flip(frame, 1)
        # Enganchamos y actualizamos el pipeline principal derivando calculos estructurales sobre el buffer recien extraido de la cola V4L2
        tracker.process_frame(frame)
        # Obligamos la recuperacion posicional inyectando ponderadores exponenciales y retornando la lista final procesada linealmente
        landmarks = tracker.get_landmarks(smooth=True)
        # Validamos estrictamente las condiciones espaciales y procedemos con invocacion en cascada hacia el volcado en el sistema de disco
        if landmarks and len(landmarks) == 63:
            if save_data(label, landmarks):
                captured += 1
        # Generamos la interfaz secundaria transitoria invocando los modificadores GDI nativos superpuestos sobre el pipeline procesado
        frame_show = tracker.draw_hands(frame)
        # Aplicamos trazado alfanumerico proyectando los identificadores escalares de progreso sobre las coordenadas inmutables del raster
        cv2.putText(frame_show, f"Auto: {label} - {captured}/{num_samples}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("NeuralSign-LSM : Data Gathering", frame_show)
        cv2.waitKey(1)
        if delay > 0:
            time.sleep(delay)
    print(f"Auto capture finished. Saved {captured} samples for {label}.\n")

def save_sequence(label, sequence):
    # Fusionamos los literales subyacentes mapeando el contenedor del lote especifico asociado a su clasificacion objetivo general
    seq_dir = os.path.join("data", "sequences", label)
    os.makedirs(seq_dir, exist_ok=True)
    # Generamos la llave cifrada basandonos en el epoch time flotante truncado a una magnitud referencial entera pseudo-unica
    timestamp = int(time.time() * 1000)
    # Compilamos la ruta destino absoluta acoplando el timestamp y fijando arbitrariamente la extension exclusiva para Numpy arrays
    filename = os.path.join(seq_dir, f"{timestamp}.npy")
    # Forzamos la serializacion estatica transcribiendo el lote anidado y transformado en formato matricial hacia disco duro secundario
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
    # Inicializamos un iterador de profundidad bloqueante capturando consecutivamente los tensores sin desviar memoria a funciones asincronas
    for i in range(seq_len):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            return
        # Reflejamos el eje horizontal para estandarizar la entrada con el flujo del modelo entrenado y evitar inversiones en X posicionales
        frame = cv2.flip(frame, 1)
        tracker.process_frame(frame)
        landmarks = tracker.get_landmarks(smooth=True)
        # Rompemos incondicionalmente el bucle abortando la secuencia actual si el marco omite caracteristicas impidiendo la concatenacion pura
        if len(landmarks) != 63:
            print(f"Frame {i+1}: hand not detected, recording cancelled")
            return
        # Insertamos ordenadamente la posicion lineal actual acrecentando el tamanio del marco espacial hasta el maximo establecido en tensor
        buffer.append(landmarks)
        frame_show = tracker.draw_hands(frame)
        cv2.putText(frame_show, f"Recording {label} - {i+1}/{seq_len}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("NeuralSign-LSM : Data Gathering", frame_show)
        cv2.waitKey(1)
        # Asignamos un intervalo diferencial para espaciar el barrido del hardware compensando la tasa de captura natural del hilo
        time.sleep(0.05)
    # Evaluamos retrospectivamente si el limite acumulativo encajo perfectamente descartando aquellos buffers huerfanos por perdida de hardware
    if len(buffer) == seq_len:
        save_sequence(label, buffer)
        print(f"Sequence for {label} saved successfully")
    else:
        print("Incomplete sequence, not saved")

def main():
    # Asignamos el canal cero montando el descriptor por defecto al bus primario conectandolo con el driver generico del hardware visual
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    # Saturamos los limites logicos del API inyectando propiedades macro limitando el margen geometrico original de la lente conectada
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Invocamos la inicializacion estableciendo la permeabilidad logica en un factor discreto controlando inercia posicional del rastreador
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
            # Solicitamos activamente la captura desencadenando operacion ioctl e importando el bloque a una estructura de despiece manejada
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            frame = cv2.flip(frame, 1)
            tracker.process_frame(frame)
            
            # Encapsulamos la representacion grafica invocando el motor y dejando el objeto matriz alterado preparado para buffer en pantalla
            frame = tracker.draw_hands(frame)
            # Mantenemos viva la estructura interna invocando un fetch paralelo hacia los escalares estabilizados antes de su alteracion inminente
            landmarks = tracker.get_landmarks(smooth=True)

            if last_detected_label:
                # Escribimos los caracteres codificados aplicando primitivas renderizadas de bajo impacto con tipografia parametrizada estaticamente
                cv2.putText(frame, f"Current letter: {last_detected_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Press a-z to choose a letter", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, "[C] 1 sample | [V] 500 samples | [B] sequence | [Q] quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Volcamos la memoria acumulada proyectando directamente contra los apuntadores API del sistema grafico base delegando escalado
            cv2.imshow("NeuralSign-LSM : Data Gathering", frame)

            # Restringimos el flujo durante el periodo umbral interrumpiendo subrutinas hasta extraer el input del socket de mascara logica
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            # Rompemos la iteracion bloqueando ejecuciones subsiguientes y entregando la instruccion de volcado final hacia los constructores
            if key == ord('Q') or key == ord('q') or key == 27:
                break

            elif key == ord('C'):
                # Sometemos los flags posicionales y etiquetas temporales validando compatibilidad de limites previos a volcar memoria
                if last_detected_label and landmarks and len(landmarks) == 63:
                    save_data(last_detected_label, landmarks)
                else:
                    print("No letter defined or hand not detected")

            elif key == ord('V'):
                # Forzamos una interrupcion de control lanzando el apuntador a la rutina de captura por batch restringiendo operaciones UI
                if last_detected_label and landmarks and len(landmarks) == 63:
                    auto_capture(tracker, cap, last_detected_label, num_samples=500, delay=0.0)
                else:
                    print("No letter defined or hand not detected")

            elif key == ord('B'):
                # Derivamos la carga del socket logico invocando el constructor secuencial pasando los identificadores para anidamiento tridimensional
                if last_detected_label:
                    record_sequence(tracker, cap, last_detected_label, seq_len=30)
                else:
                    print("First define a letter by pressing a-z")

            elif (97 <= key <= 122) or (65 <= key <= 90):
                # Formateamos el ordinal basico ASCII invirtiendo mayusculas hacia el apuntador transaccional que dictamina la clase global
                last_detected_label = chr(key).upper()
                print(f"[*] Ready to record letter: {last_detected_label}")

            else:
                # Omitimos el proceso silenciosamente preservando overhead del nucleo devolviendo control al despachador sin ejecutar cambios logicos
                pass

    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        # Purgamos obligatoriamente el vinculo con el manejador de video restaurando dependencias en el kernel hacia controladores locales
        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":
    main()