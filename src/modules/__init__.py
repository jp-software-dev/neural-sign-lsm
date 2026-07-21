# Inicializa el paquete 'modules' y expone sus componentes principales.
from .game_logic import GameMode, SignGame
from .voice_bot import VoiceAssistant
from .word_bank import get_words_by_difficulty, get_all_words, WORD_BANK
from . import profile_manager, sound_manager