# Hace que el directorio 'modules' sea un paquete de Python y expone sus componentes.
# Esto permite importaciones más limpias como 'from src.modules import VoiceAssistant'.
from .game_logic import GameMode, SignGame
from .voice_bot import VoiceAssistant
from .word_bank import get_words_by_difficulty, get_all_words, WORD_BANK
from . import profile_manager