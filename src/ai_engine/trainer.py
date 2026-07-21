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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras import regularizers # type: ignore

from src.config.settings import (
    LANDMARKS_CSV_PATH, AI_MODEL_PATH, LABELS_JSON_PATH,
    MODELS_DIR, SCALER_PATH,
)
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

def set_seed(seed: int = 42):
    # Establece una semilla fija para bibliotecas de aleatoriedad para asegurar la reproducibilidad.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_seed(42)

class SignLanguageTrainer:
    # Orquesta el proceso de entrenamiento del modelo de IA.
    # Soporta modos 'static' (DNN) y 'lstm' (red recurrente).
    def __init__(self, mode: str = 'static', seq_len: int = 30):
        self.mode = mode
        self.seq_len = seq_len
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.num_classes = 0
        self._y_encoded_train: np.ndarray | None = None

    def load_data(self):
        # Carga los datos según el modo de entrenamiento (estático o secuencial).
        if self.mode == 'static':
            return self._load_static_data()
        return self._load_sequence_data()

    def _load_static_data(self):
        # Carga y preprocesa los datos desde el archivo CSV para el modelo estático.
        try:
            df = pd.read_csv(LANDMARKS_CSV_PATH, header=None)
            y_raw = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values.astype(np.float32)
            X = self.scaler.fit_transform(X)
            self._save_scaler()
            y_encoded = self.label_encoder.fit_transform(y_raw)
            self.num_classes = len(self.label_encoder.classes_)
            y = to_categorical(y_encoded, num_classes=self.num_classes)
            self.save_labels()
            self._y_encoded_all = y_encoded
            return X, y
        except Exception as e:
            app_logger.error(f"Error cargando dataset estático: {str(e)}")
            raise

    def _load_sequence_data(self):
        # Carga y preprocesa los datos desde archivos .npy para el modelo LSTM.
        try:
            sequences, labels = [], []
            base_dir = "data/sequences"
            if not os.path.exists(base_dir):
                raise FileNotFoundError(f"Directorio {base_dir} no encontrado.")
            for letra in os.listdir(base_dir):
                letra_path = os.path.join(base_dir, letra)
                if not os.path.isdir(letra_path):
                    continue
                for file in os.listdir(letra_path):
                    if file.endswith('.npy'):
                        seq = np.load(os.path.join(letra_path, file))
                        if seq.shape != (self.seq_len, 63):
                            continue
                        sequences.append(seq)
                        labels.append(letra)
            if not sequences:
                raise ValueError("No se encontraron secuencias válidas.")
            X = np.array(sequences)
            X_reshaped = X.reshape(-1, 63)
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(-1, self.seq_len, 63)
            self._save_scaler()
            y_raw = np.array(labels)
            y_encoded = self.label_encoder.fit_transform(y_raw)
            self.num_classes = len(self.label_encoder.classes_)
            y = to_categorical(y_encoded, num_classes=self.num_classes)
            self.save_labels()
            self._y_encoded_all = y_encoded
            return X, y
        except Exception as e:
            app_logger.error(f"Error cargando dataset secuencial: {str(e)}")
            raise

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

    def _augment_data(self, X, y, noise_factor=0.04):
        # Aplica aumento de datos añadiendo ruido gaussiano.
        noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
        X_noisy = X + noise
        X_combined = np.vstack((X, X_noisy))
        y_combined = np.vstack((y, y))
        return X_combined, y_combined

    def train_model(self):
        # Función principal que ejecuta todo el pipeline de entrenamiento.
        try:
            X, y = self.load_data()
            X_train, X_test, y_train, y_test, idx_train, _ = train_test_split(
                X, y, np.arange(len(y)), test_size=0.2, random_state=42
            )
            
            # Aplica aumento de datos solo al conjunto de entrenamiento
            if self.mode == 'static':
                X_train, y_train = self._augment_data(X_train, y_train)

            if self.mode == 'static':
                self.build_model(input_shape=X_train.shape[1])
            else:
                self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                
            # Calcula pesos de clases para manejar datasets desbalanceados
            y_encoded_train = np.argmax(y_train, axis=1)
            classes = np.unique(y_encoded_train)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_encoded_train,
            )
            class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
            
            # Define callbacks para el entrenamiento
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            checkpoint = ModelCheckpoint(AI_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=120,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=[early_stop, checkpoint, reduce_lr],
            )
            # Genera y guarda el reporte de evaluación final
            self._save_training_report(X_test, y_test)
        except Exception as e:
            app_logger.error(f"Fallo crítico durante el entrenamiento: {e}")

    def _save_training_report(self, X_test, y_test):
        # Evalúa el modelo con los datos de prueba y guarda un reporte de clasificación,
        # una matriz de confusión y visualizaciones de errores.
        try:
            print("\n[INFO] Generando reporte de clasificación y matriz de confusión...")
            y_pred_probs = self.model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_test, axis=1)
            report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, zero_division=0)
            conf_matrix = confusion_matrix(y_true, y_pred)
            report_path = os.path.join(MODELS_DIR, "training_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Reporte de Clasificación del Entrenamiento\n")
                f.write("="*40 + "\n")
                f.write(report)
                f.write("\n\nMatriz de Confusión\n")
                f.write("="*40 + "\n")
                f.write(np.array2string(conf_matrix, separator=', '))
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
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            app_logger.error(f"Error guardando el escalador: {str(e)}")

    def save_labels(self):
        # Guarda el mapeo de índices a etiquetas de clase en un archivo JSON.
        label_mapping = {int(i): str(label) for i, label in enumerate(self.label_encoder.classes_)}
        try:
            with open(LABELS_JSON_PATH, 'w') as f:
                json.dump(label_mapping, f, indent=4)
        except Exception as e:
            app_logger.error(f"Error guardando JSON de etiquetas: {str(e)}")

if __name__ == "__main__":
    # --- Bloque de Ejecución Principal ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'lstm'])
    parser.add_argument('--seq_len', type=int, default=30)
    args = parser.parse_args()
    trainer = SignLanguageTrainer(mode=args.mode, seq_len=args.seq_len)
    trainer.train_model()