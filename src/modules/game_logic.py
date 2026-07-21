"""Define la arquitectura y lógica de los modos de juego."""
import random
import time
from enum import Enum, auto

from src.utils import app_logger
from src.modules.word_bank import get_words_by_difficulty

class GameMode(Enum):
    """Enumera los diferentes modos de juego disponibles."""
    NONE = auto()
    GAME = auto()

class BaseGame:
    """Define la estructura y comportamiento base para todos los juegos."""

    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        self.available_letters = [l.upper() for l in available_letters]
        self.difficulty = difficulty

        # --- Estado del Juego ---
        self.score: int = 0
        self.game_active: bool = False
        self.start_time: float = 0.0
        self.time_limit: int = 60

        # --- Estado de la Palabra Actual ---
        self.target_word: str = ""
        self.target_letter: str = ""
        self._letter_index: int = 0
        self._words_completed: int = 0
        self._word_pool: list[str] = []

    def start_game(self, difficulty: str | None = None):
        """Prepara e inicia una nueva sesión de juego."""
        if difficulty:
            self.difficulty = difficulty

        self.score = 0
        self._words_completed = 0
        self.game_active = True
        self.start_time = time.time()

        self._build_word_pool()
        self._pick_new_word()

    def check_time(self) -> tuple[bool, str]:
        """Verifica si el tiempo límite de la partida ha sido alcanzado."""
        if not self.game_active:
            return False, ""
        if self.time_left <= 0:
            self.game_active = False
            return True, f"¡Tiempo! Puntuación: {self.score} | Palabras: {self._words_completed}"
        return False, ""

    def check_prediction(self, predicted_letter: str) -> tuple[bool, str]:
        """Procesa una predicción de seña. Debe ser implementado por subclases."""
        raise NotImplementedError("Las subclases deben implementar check_prediction.")

    @property
    def time_left(self) -> int:
        """Calcula y devuelve el tiempo restante de la partida."""
        if not self.game_active:
            return 0
        return max(0, int(self.time_limit - (time.time() - self.start_time)))

    @property
    def progress_display(self) -> str:
        """Devuelve un string que representa el progreso en la palabra actual."""
        return self._build_progress_display()

    @property
    def words_completed(self) -> int:
        """Devuelve el contador de palabras completadas."""
        return self._words_completed

    def _build_word_pool(self):
        """Construye el banco de palabras para la partida. Debe ser implementado."""
        raise NotImplementedError("Las subclases deben implementar _build_word_pool.")

    def _pick_new_word(self):
        """Selecciona una nueva palabra objetivo del banco de palabras."""
        self.target_word = random.choice(self._word_pool).upper()
        self._letter_index = 0
        self.target_letter = self.target_word[0]

    def _advance_letter(self) -> bool:
        """Avanza el objetivo a la siguiente letra de la palabra actual."""
        self._letter_index += 1
        if self._letter_index >= len(self.target_word):
            return True
        self.target_letter = self.target_word[self._letter_index]
        return False

    def _build_progress_display(self) -> str:
        """Genera el string visual del progreso (ej: [H] O L A)."""
        chars = []
        for i, ch in enumerate(self.target_word):
            if i < self._letter_index:
                chars.append(ch)
            elif i == self._letter_index:
                chars.append(f"[{ch}]")
            else:
                chars.append("_")
        return " ".join(chars)


class SignGame(BaseGame):
    """Implementa el modo de juego principal basado en tiempo, puntos y rachas."""
    MODE = GameMode.GAME

    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        super().__init__(available_letters, difficulty)
        # --- Estado Específico del Juego ---
        self.streak: int = 0
        self.peak_streak: int = 0
        self.time_limit: int = 60

    def start_game(self, difficulty: str | None = None) -> str:
        """Inicia una nueva partida de SignGame."""
        super().start_game(difficulty)
        self.streak = 0
        self.peak_streak = 0
        return self.target_letter

    def check_time(self) -> tuple[bool, str]:
        """Verifica el tiempo y finaliza el juego si se ha agotado."""
        if not self.game_active:
            return False, ""
        if self.time_left <= 0:
            self.game_active = False
            return True, f"¡Tiempo! Puntuación: {self.score} pts | Palabras: {self._words_completed}"
        return False, ""

    def check_prediction(self, predicted_letter: str) -> tuple[bool, str]:
        """Procesa la predicción del usuario y actualiza el estado del juego."""
        if not self.game_active:
            return False, "Juego terminado"

        expired, msg = self.check_time()
        if expired:
            return False, msg

        if predicted_letter.upper() == self.target_letter:
            self.streak += 1
            if self.streak > self.peak_streak:
                self.peak_streak = self.streak          
            multiplier = min(self.streak, 5)
            points = 10 * multiplier
            self.score += points

            word_done = self._advance_letter()

            if word_done:
                self._words_completed += 1
                self._pick_new_word()
                return True, f"¡Palabra! +{points}pts (×{multiplier}) | Total: {self.score}"
            else:
                return True, f"¡Bien! +{points}pts (×{multiplier}) | Sig: {self.target_letter}"
        else:
            self.streak = 0
            progress = self._build_progress_display()
            return False, f"{progress} | Pts:{self.score} | {self.time_left}s"

    def _build_word_pool(self):
        """Construye el banco de palabras filtrando por las letras disponibles."""
        raw = get_words_by_difficulty(self.difficulty)
        self._word_pool = [
            w for w in raw
            if all(ch in self.available_letters for ch in w.upper())
        ]
        if not self._word_pool:
            self._word_pool = self.available_letters[:10] or ["A"]