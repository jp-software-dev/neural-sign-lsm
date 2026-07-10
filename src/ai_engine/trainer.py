import os
import json
import argparse
import pickle
import pandas as pd
import numpy as np
import random
import tensorflow as tf # type: ignore
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

def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_seed(42)

class SignLanguageTrainer:
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
        if self.mode == 'static':
            return self._load_static_data()
        return self._load_sequence_data()

    def _load_static_data(self):
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
        if self.mode == 'static':
            self._build_static_model(input_shape)
        else:
            self._build_lstm_model(input_shape)

    def _build_static_model(self, input_shape):
        self.model = Sequential([
            InputLayer(input_shape=(input_shape,)),
            Dense(256, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dense(self.num_classes, activation='softmax'),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def _build_lstm_model(self, input_shape):
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

    def train_model(self):
        try:
            X, y = self.load_data()
            X_train, X_test, y_train, y_test, idx_train, _ = train_test_split(
                X, y, np.arange(len(y)), test_size=0.2, random_state=42
            )
            if self.mode == 'static':
                self.build_model(input_shape=X.shape[1])
            else:
                self.build_model(input_shape=(X.shape[1], X.shape[2]))
            y_encoded_train = self._y_encoded_all[idx_train]
            classes = np.unique(y_encoded_train)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_encoded_train,
            )
            class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            checkpoint = ModelCheckpoint(AI_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
            self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=[early_stop, checkpoint, reduce_lr],
            )
        except Exception as e:
            app_logger.error(f"Fallo crítico durante el entrenamiento: {str(e)}")

    def _save_scaler(self):
        try:
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            app_logger.error(f"Error guardando el escalador: {str(e)}")

    def save_labels(self):
        label_mapping = {int(i): str(label) for i, label in enumerate(self.label_encoder.classes_)}
        try:
            with open(LABELS_JSON_PATH, 'w') as f:
                json.dump(label_mapping, f, indent=4)
        except Exception as e:
            app_logger.error(f"Error guardando JSON de etiquetas: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'lstm'])
    parser.add_argument('--seq_len', type=int, default=30)
    args = parser.parse_args()
    trainer = SignLanguageTrainer(mode=args.mode, seq_len=args.seq_len)
    trainer.train_model()