# Script para entrenar el modelo de reconocimiento de señas (estático o LSTM).
# Incluye carga de datos, construcción del modelo, entrenamiento y evaluación.
import os
import json
import argparse
import pickle
import pandas as pd
import numpy as np
import random
import cv2
import tensorflow as tf # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, Dropout, BatchNormalization,
    LSTM, InputLayer, Activation,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras import regularizers # type: ignore

from src.config.settings import (
    LANDMARKS_CSV_PATH, AI_MODEL_PATH, LABELS_JSON_PATH,
    MODELS_DIR, SCALER_PATH, SEQUENCES_DIR,
    LSTM_MODEL_PATH, LSTM_LABELS_JSON_PATH, LSTM_SCALER_PATH,
)
from src.ai_engine.features import normalize_landmarks
from src.utils import app_logger

# --- Constantes para Dibujo ---
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

# Las capturas automáticas guardan ráfagas de 500 frames consecutivos; cada ráfaga se trata
# como un grupo para que la validación use grabaciones que el modelo no vio al entrenar.
CAPTURE_GROUP_SIZE = 500

# --- Funciones Auxiliares ---
def _draw_landmarks_on_canvas(landmarks_flat, true_label, pred_label, img_size=400, padding=40):
    # Dibuja los landmarks de una mano en una imagen en blanco.
    # Utilizado para visualizar los errores de predicción del modelo.
    if landmarks_flat is None or len(landmarks_flat) != 63:
        return None

    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    pts = np.array(landmarks_flat).reshape(21, 3)

    # Normaliza y centra la mano en el lienzo
    x_coords, y_coords = pts[:, 0], pts[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    scale = min(
        (img_size - 2 * padding) / (max_x - min_x) if (max_x - min_x) > 0 else 1,
        (img_size - 2 * padding) / (max_y - min_y) if (max_y - min_y) > 0 else 1
    )
    offset_x = padding - min_x * scale
    offset_y = padding - min_y * scale

    pixel_pts = [(int(p[0] * scale + offset_x), int(p[1] * scale + offset_y)) for p in pts]

    # Dibuja las conexiones y los puntos
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(img, pixel_pts[a], pixel_pts[b], (220, 220, 220), 1, cv2.LINE_AA)
    for idx, pt in enumerate(pixel_pts):
        radius = 6 if idx in (4, 8, 12, 16, 20) else 4
        cv2.circle(img, pt, radius, (0, 255, 120), -1, cv2.LINE_AA)

    # Agrega texto de etiquetas
    cv2.putText(img, f"Real: {true_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img, f"Prediccion: {pred_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img

def capture_groups(labels):
    # Asigna un id de grupo a cada fila: ráfagas contiguas de la misma letra, partidas cada 500 filas.
    labels = pd.Series(labels)
    run_id = (labels != labels.shift()).cumsum().values
    pos_in_run = pd.Series(np.arange(len(labels))).groupby(run_id).cumcount().values
    return run_id * 1000 + pos_in_run // CAPTURE_GROUP_SIZE

def augment_hands(X, rng, noise_factor=0.02, max_rotation=0.25, scale_range=(0.9, 1.1)):
    # Genera copias con rotación en el plano, cambio de escala y ruido gaussiano,
    # simulando variaciones naturales de la mano frente a la cámara.
    pts = X.reshape(-1, 21, 3).copy()
    angles = rng.uniform(-max_rotation, max_rotation, len(pts))
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x, y = pts[..., 0].copy(), pts[..., 1].copy()
    pts[..., 0] = cos * x - sin * y
    pts[..., 1] = sin * x + cos * y
    pts *= rng.uniform(*scale_range, (len(pts), 1, 1))
    pts += rng.normal(0.0, noise_factor, pts.shape)
    return pts.reshape(-1, 63).astype(np.float32)

def set_seed(seed: int = 42):
    # Establece una semilla fija para bibliotecas de aleatoriedad para asegurar la reproducibilidad.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

class SignLanguageTrainer:
    # Orquesta el proceso de entrenamiento del modelo de IA.
    # Soporta modos 'static' (DNN) y 'lstm' (red recurrente).
    def __init__(self, mode: str = 'static', seq_len: int = 30):
        self.mode = mode
        self.seq_len = seq_len
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        if mode == 'static':
            self.model_path, self.labels_path, self.scaler_path = AI_MODEL_PATH, LABELS_JSON_PATH, SCALER_PATH
        else:
            self.model_path, self.labels_path, self.scaler_path = LSTM_MODEL_PATH, LSTM_LABELS_JSON_PATH, LSTM_SCALER_PATH
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.num_classes = 0
        self.rng = np.random.default_rng(42)

    def load_data(self):
        # Carga los datos según el modo de entrenamiento (estático o secuencial).
        # Devuelve features normalizadas, etiquetas codificadas y grupos de captura.
        if self.mode == 'static':
            return self._load_static_data()
        return self._load_sequence_data()

    def _load_static_data(self):
        # Carga el CSV de landmarks y los normaliza respecto a la muñeca y el tamaño de la mano.
        try:
            df = pd.read_csv(LANDMARKS_CSV_PATH, header=None)
            y_raw = df.iloc[:, 0].values
            X = normalize_landmarks(df.iloc[:, 1:].values.astype(np.float32))
            y = self.label_encoder.fit_transform(y_raw)
            self.num_classes = len(self.label_encoder.classes_)
            self.save_labels()
            return X, y, capture_groups(y_raw)
        except Exception as e:
            app_logger.error(f"Error cargando dataset estático: {str(e)}")
            raise

    def _load_sequence_data(self):
        # Carga y normaliza las secuencias .npy para el modelo LSTM.
        try:
            sequences, labels = [], []
            if not os.path.exists(SEQUENCES_DIR):
                raise FileNotFoundError(f"Directorio {SEQUENCES_DIR} no encontrado.")
            for letra in os.listdir(SEQUENCES_DIR):
                letra_path = os.path.join(SEQUENCES_DIR, letra)
                if not os.path.isdir(letra_path):
                    continue
                for file in os.listdir(letra_path):
                    if file.endswith('.npy'):
                        seq = np.load(os.path.join(letra_path, file))
                        if seq.shape != (self.seq_len, 63):
                            continue
                        sequences.append(normalize_landmarks(seq))
                        labels.append(letra)
            if not sequences:
                raise ValueError("No se encontraron secuencias válidas.")
            X = np.array(sequences, dtype=np.float32)
            y = self.label_encoder.fit_transform(np.array(labels))
            self.num_classes = len(self.label_encoder.classes_)
            self.save_labels()
            # Cada secuencia es una grabación independiente
            return X, y, np.arange(len(y))
        except Exception as e:
            app_logger.error(f"Error cargando dataset secuencial: {str(e)}")
            raise

    def _split(self, X, y, groups):
        # Estático: la última ráfaga grabada de cada letra se reserva para validación.
        # LSTM: división aleatoria estratificada (pocas secuencias por letra).
        if self.mode == 'static':
            test = np.zeros(len(y), dtype=bool)
            for cls in np.unique(y):
                cls_groups = np.unique(groups[y == cls])
                if len(cls_groups) > 1:
                    test |= groups == cls_groups.max()
            return ~test, test
        idx_train, idx_test = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
        )
        train = np.zeros(len(y), dtype=bool)
        train[idx_train] = True
        return train, ~train

    def _fit_scaler(self, X):
        flat = X.reshape(-1, 63)
        self.scaler = StandardScaler().fit(flat)

    def _scale(self, X):
        return self.scaler.transform(X.reshape(-1, 63)).reshape(X.shape).astype(np.float32)

    def build_model(self, input_shape):
        # Construye la arquitectura del modelo (DNN o LSTM) según el modo.
        if self.mode == 'static':
            self._build_static_model(input_shape)
        else:
            self._build_lstm_model(input_shape)

    def _build_static_model(self, input_shape):
        # Define la arquitectura de la Red Neuronal Densa (DNN) para señas estáticas.
        self.model = Sequential([
            InputLayer(input_shape=(input_shape,)),
            Dense(256, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.35),
            Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dense(self.num_classes, activation='softmax'),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def _build_lstm_model(self, input_shape):
        # Define la arquitectura de la Red Neuronal Recurrente (LSTM) para secuencias.
        self.model = Sequential([
            InputLayer(input_shape=input_shape),
            LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),
            LSTM(128, kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),
            Dense(64),
            BatchNormalization(),
            Activation('relu'),
            Dense(self.num_classes, activation='softmax'),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def _augment_data(self, X, y):
        # Duplica el conjunto de entrenamiento con variaciones geométricas (solo modo estático).
        X_aug = augment_hands(X, self.rng)
        return np.vstack((X, X_aug)), np.concatenate((y, y))

    def _fit(self, X_train, y_train, X_val=None, y_val=None, epochs=120):
        # Entrena un modelo nuevo; con datos de validación usa early stopping.
        if self.mode == 'static':
            X_train, y_train = self._augment_data(X_train, y_train)
            self.build_model(input_shape=X_train.shape[1])
        else:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Calcula pesos de clases para manejar datasets desbalanceados
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

        validation_data, callbacks = None, []
        monitor = 'val_loss' if X_val is not None else 'loss'
        if X_val is not None:
            validation_data = (X_val, to_categorical(y_val, num_classes=self.num_classes))
            callbacks.append(EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True))
        callbacks.append(ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=1e-5, verbose=1))

        history = self.model.fit(
            X_train, to_categorical(y_train, num_classes=self.num_classes),
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=callbacks,
        )
        return history

    def train_model(self, final_fit: bool = True):
        # Función principal que ejecuta todo el pipeline de entrenamiento.
        # 1) Entrena y evalúa con grabaciones no vistas. 2) Reentrena con todos los datos
        #    durante el número de épocas óptimo encontrado y guarda ese modelo.
        try:
            set_seed(42)
            X, y, groups = self.load_data()
            train_mask, test_mask = self._split(X, y, groups)
            X_train, y_train, X_test, y_test = X[train_mask], y[train_mask], X[test_mask], y[test_mask]

            self._fit_scaler(X_train)
            history = self._fit(self._scale(X_train), y_train, self._scale(X_test), y_test)
            best_epoch = int(np.argmin(history.history['val_loss'])) + 1
            self._save_training_report(self._scale(X_test), y_test, best_epoch)

            if final_fit:
                print(f"\n[INFO] Reentrenando con todos los datos durante {best_epoch} épocas...")
                set_seed(42)
                self._fit_scaler(X)
                self._fit(self._scale(X), y, epochs=best_epoch)

            self.model.save(self.model_path)
            self._save_scaler()
            print(f"[INFO] Modelo guardado en: {self.model_path}")
        except Exception as e:
            app_logger.error(f"Fallo crítico durante el entrenamiento: {e}")
            raise

    def _save_training_report(self, X_test, y_test, best_epoch):
        # Evalúa el modelo con los datos de validación y guarda un reporte de clasificación,
        # una matriz de confusión y visualizaciones de errores.
        try:
            print("\n[INFO] Generando reporte de clasificación y matriz de confusión...")
            y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
            y_true = y_test
            labels_idx = np.arange(self.num_classes)
            report = classification_report(
                y_true, y_pred, labels=labels_idx,
                target_names=self.label_encoder.classes_, zero_division=0,
            )
            conf_matrix = confusion_matrix(y_true, y_pred, labels=labels_idx)
            accuracy = float(np.mean(y_pred == y_true))
            report_path = os.path.join(MODELS_DIR, f"training_report_{self.mode}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Reporte de Clasificación del Entrenamiento\n")
                f.write("="*40 + "\n")
                f.write(f"Validación: grabaciones no vistas en el entrenamiento ({len(y_true)} muestras)\n")
                f.write(f"Precisión de validación: {accuracy:.4f}\n")
                f.write(f"Mejor época: {best_epoch}\n\n")
                f.write(report)
                f.write("\n\nMatriz de Confusión\n")
                f.write("="*40 + "\n")
                f.write(np.array2string(conf_matrix, separator=', ', max_line_width=200))
            print(f"[INFO] Precisión de validación: {accuracy:.4f}")
            print(f"[INFO] Reporte guardado en: {report_path}")

            # Guarda imágenes de algunas predicciones incorrectas para análisis visual
            print("[INFO] Visualizando predicciones incorrectas...")
            incorrect_indices = np.where(y_pred != y_true)[0]

            if len(incorrect_indices) == 0:
                print("[INFO] No se encontraron predicciones incorrectas para visualizar.")
                return

            error_dir = os.path.join(MODELS_DIR, "error_analysis")
            os.makedirs(error_dir, exist_ok=True)

            num_to_show = min(5, len(incorrect_indices))
            show_indices = random.sample(list(incorrect_indices), num_to_show)

            for i, idx in enumerate(show_indices):
                landmarks_scaled = X_test[idx]
                if self.mode == 'lstm':
                    landmarks_scaled = landmarks_scaled[-1]
                landmarks_flat = self.scaler.inverse_transform(landmarks_scaled.reshape(1, -1))[0]
                true_label = self.label_encoder.classes_[y_true[idx]]
                pred_label = self.label_encoder.classes_[y_pred[idx]]
                error_image = _draw_landmarks_on_canvas(landmarks_flat, true_label, pred_label)
                if error_image is not None:
                    filename = f"error_{i+1}_true-{true_label}_pred-{pred_label}.png"
                    cv2.imwrite(os.path.join(error_dir, filename), error_image)
            print(f"[INFO] {len(show_indices)} imágenes de error guardadas en: {error_dir}")
        except Exception as e:
            app_logger.error(f"No se pudo generar el reporte de entrenamiento o visualizar errores: {e}")

    def _save_scaler(self):
        # Guarda el objeto `StandardScaler` ajustado en un archivo pickle.
        try:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            app_logger.error(f"Error guardando el escalador: {str(e)}")

    def save_labels(self):
        # Guarda el mapeo de índices a etiquetas de clase en un archivo JSON.
        label_mapping = {int(i): str(label) for i, label in enumerate(self.label_encoder.classes_)}
        try:
            with open(self.labels_path, 'w') as f:
                json.dump(label_mapping, f, indent=4)
        except Exception as e:
            app_logger.error(f"Error guardando JSON de etiquetas: {str(e)}")

if __name__ == "__main__":
    # --- Bloque de Ejecución Principal ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'lstm'])
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--no-final-fit', action='store_true',
                        help='Guarda el modelo validado sin reentrenar con todos los datos')
    args = parser.parse_args()
    trainer = SignLanguageTrainer(mode=args.mode, seq_len=args.seq_len)
    trainer.train_model(final_fit=not args.no_final_fit)
