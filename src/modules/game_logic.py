"""
game_logic.py
-------------
Motor de gamificación del sistema NeuralSign-LSM.

Mejoras respecto a la versión anterior:
  - Palabras reales del banco (word_bank.py) en lugar de letras sueltas
  - Progresión letra a letra dentro de la palabra objetivo
  - Sistema de racha (streak) que multiplica los puntos
  - Soporte de dificultad: facil / medio / dificil
  - Logger corregido a nivel ERROR (el nivel INFO no se guardaba)
"""

import random
import time

from src.utils import app_logger
from src.modules.word_bank import get_words_by_difficulty, get_all_words


class SignGame:
    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        # Corpus de letras que el modelo es capaz de clasificar (filtro de seguridad)
        self.available_letters = [l.upper() for l in available_letters]

        # Dificultad de la sesión: determina el banco de palabras a usar
        self.difficulty = difficulty

        # ── Estado de sesión ──────────────────────────────────────────
        self.score = 0
        self.streak = 0          # Aciertos consecutivos sin error
        self.game_active = False
        self.start_time = 0.0
        self.time_limit = 60     # segundos por defecto

        # ── Estado de la palabra actual ───────────────────────────────
        self.target_word = ""          # Palabra completa que el usuario debe formar
        self.target_letter = ""        # Letra actual a mostrar
        self._letter_index = 0         # Posición dentro de target_word
        self._words_completed = 0      # Palabras completadas en la sesión

        # ── Banco de palabras filtrado ────────────────────────────────
        self._word_pool: list[str] = []

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def start_game(self, difficulty: str | None = None) -> str:
        """
        Inicia una nueva sesión de juego.
        Retorna la primera letra objetivo para que main.py la anuncie.
        """
        if difficulty:
            self.difficulty = difficulty

        # Reseteamos contadores de sesión
        self.score = 0
        self.streak = 0
        self._words_completed = 0
        self.game_active = True
        self.start_time = time.time()

        # Construimos el banco filtrando solo palabras cuyas letras el modelo conoce
        raw_pool = get_words_by_difficulty(self.difficulty)
        self._word_pool = [
            w for w in raw_pool
            if all(ch in self.available_letters for ch in w.upper())
        ]

        # Fallback: si ninguna palabra pasa el filtro usamos letras sueltas
        if not self._word_pool:
            self._word_pool = self.available_letters

        self._pick_new_word()
        return self.target_letter

    def check_prediction(self, predicted_letter: str) -> tuple[bool, str]:
        """
        Evalúa si la letra predicha coincide con la letra objetivo actual.

        Retorna:
            (is_correct: bool, status_text: str)
        """
        if not self.game_active:
            return False, "Juego terminado"

        elapsed = time.time() - self.start_time
        time_left = int(self.time_limit - elapsed)

        if time_left <= 0:
            self.game_active = False
            return False, f"¡Tiempo! Puntuación final: {self.score} pts | Palabras: {self._words_completed}"

        if predicted_letter.upper() == self.target_letter:
            # ── Acierto ──────────────────────────────────────────────
            self.streak += 1
            multiplier = min(self.streak, 5)   # máximo ×5
            points = 10 * multiplier
            self.score += points

            word_done = self._advance_letter()

            if word_done:
                self._words_completed += 1
                self._pick_new_word()
                return True, f"¡Palabra completa! +{points} pts (×{multiplier}) | Total: {self.score}"
            else:
                return True, f"¡Correcto! +{points} pts (×{multiplier}) | Siguiente: {self.target_letter}"
        else:
            # ── Error: reiniciamos racha ──────────────────────────────
            self.streak = 0
            progress = self._build_progress_display()
            return False, (
                f"Forma: {progress} | "
                f"Pts: {self.score} | "
                f"Tiempo: {time_left}s"
            )

    def pick_new_letter(self) -> str:
        """
        Compatibilidad con main.py: avanza a la siguiente letra/palabra
        y retorna la nueva letra objetivo.
        """
        done = self._advance_letter()
        if done:
            self._words_completed += 1
            self._pick_new_word()
        return self.target_letter

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _pick_new_word(self):
        """Selecciona aleatoriamente una nueva palabra y reinicia el índice."""
        if self._word_pool:
            self.target_word = random.choice(self._word_pool).upper()
        else:
            self.target_word = random.choice(self.available_letters)
        self._letter_index = 0
        self.target_letter = self.target_word[0]

    def _advance_letter(self) -> bool:
        """
        Avanza al siguiente carácter de la palabra.
        Retorna True si la palabra quedó completada.
        """
        self._letter_index += 1
        if self._letter_index >= len(self.target_word):
            return True
        self.target_letter = self.target_word[self._letter_index]
        return False

    def _build_progress_display(self) -> str:
        """
        Construye una cadena tipo  'M A _ _ _'  mostrando las letras
        ya acertadas y guiones bajos para las pendientes.
        """
        chars = []
        for i, ch in enumerate(self.target_word):
            if i < self._letter_index:
                chars.append(ch)       # ya acertada
            elif i == self._letter_index:
                chars.append(ch)       # letra actual (resaltada en UI)
            else:
                chars.append("_")      # pendiente
        return " ".join(chars)

    # ------------------------------------------------------------------
    # Propiedades de solo lectura útiles para la UI
    # ------------------------------------------------------------------

    @property
    def time_left(self) -> int:
        if not self.game_active:
            return 0
        return max(0, int(self.time_limit - (time.time() - self.start_time)))

    @property
    def progress_display(self) -> str:
        return self._build_progress_display()

    @property
    def words_completed(self) -> int:
        return self._words_completed