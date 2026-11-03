# Gestiona la carga y reproducción de efectos de sonido.
import os
from src.config.settings import SOUNDS_DIR

# pygame es opcional: si no está instalado, la aplicación funciona sin efectos de sonido
try:
    import pygame
except ImportError:
    pygame = None

sound_files = {
    "tick": "tick.wav",
    "finish": "finish.wav",
    "correct": "correct.wav",
    "incorrect": "incorrect.wav"
}

sounds = {}

def init():
    # Inicializa el mezclador de pygame y carga los sonidos.
    if pygame is None:
        print("[WARN] pygame no está instalado; se desactivan los efectos de sonido.")
        return
    try:
        pygame.mixer.init()
        for name, filename in sound_files.items():
            path = os.path.join(SOUNDS_DIR, filename)
            if os.path.exists(path):
                sounds[name] = pygame.mixer.Sound(path)
            else:
                print(f"[WARN] Archivo de sonido no encontrado: {path}")
        print("[INFO] Sound manager inicializado.")
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar pygame mixer: {e}")

def play(sound_name):
    # Reproduce un sonido si está cargado.
    if sound_name in sounds:
        try:
            sounds[sound_name].play()
        except Exception as e:
            print(f"[ERROR] No se pudo reproducir el sonido '{sound_name}': {e}")