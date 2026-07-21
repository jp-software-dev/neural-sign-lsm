import random
import time
from enum import Enum, auto

from src.utils import app_logger
from src.modules.word_bank import (
    get_words_by_difficulty
)

# Definimos una estructura enumerada de tipado estricto para gestionar el enrutamiento contextual del ciclo de vida en la capa de orquestación principal
class GameMode(Enum):
    NONE = auto()
    GAME = auto()

class BaseGame:
    """Clase base para todos los modos de juego."""
    
    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        self.available_letters = [l.upper() for l in available_letters]
        self.difficulty = difficulty
        
        # Estado del juego
        self.score: int = 0
        self.game_active: bool = False
        self.start_time: float = 0.0
        self.time_limit: int = 60  # Default, puede ser sobreescrito

        # Estado de la palabra
        self.target_word: str = ""
        self.target_letter: str = ""
        self._letter_index: int = 0
        self._words_completed: int = 0
        self._word_pool: list[str] = []

    def start_game(self, difficulty: str | None = None):
        """Lógica base para iniciar o reiniciar un juego."""
        if difficulty:
            self.difficulty = difficulty

        self.score = 0
        self._words_completed = 0
        self.game_active = True
        self.start_time = time.time()

        self._build_word_pool()
        self._pick_new_word()

    def check_time(self) -> tuple[bool, str]:
        """Verifica si el tiempo de juego ha expirado."""
        if not self.game_active:
            return False, ""
        if self.time_left <= 0:
            self.game_active = False
            return True, f"¡Tiempo! Puntuación: {self.score} | Palabras: {self._words_completed}"
        return False, ""

    def check_prediction(self, predicted_letter: str) -> tuple[bool, str]:
        """Método abstracto para ser implementado por las subclases."""
        raise NotImplementedError("Las subclases deben implementar check_prediction.")

    @property
    def time_left(self) -> int:
        """Calcula el tiempo restante en segundos."""
        if not self.game_active:
            return 0
        return max(0, int(self.time_limit - (time.time() - self.start_time)))

    @property
    def progress_display(self) -> str:
        """Genera la cadena de texto que muestra el progreso en la palabra actual."""
        return self._build_progress_display()

    @property
    def words_completed(self) -> int:
        """Retorna el número de palabras completadas."""
        return self._words_completed

    def _build_word_pool(self):
        """Método abstracto para construir el banco de palabras."""
        raise NotImplementedError("Las subclases deben implementar _build_word_pool.")

    def _pick_new_word(self):
        """Elige una nueva palabra del pool y resetea el índice."""
        self.target_word = random.choice(self._word_pool).upper()
        self._letter_index = 0
        self.target_letter = self.target_word[0]

    def _advance_letter(self) -> bool:
        """Avanza a la siguiente letra de la palabra. Retorna True si la palabra terminó."""
        self._letter_index += 1
        if self._letter_index >= len(self.target_word):
            return True
        self.target_letter = self.target_word[self._letter_index]
        return False

    def _build_progress_display(self) -> str:
        """Construye la representación visual del progreso de la palabra."""
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
    # Vinculamos la clase al identificador de enrutamiento estático correspondiente
    MODE = GameMode.GAME

    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        super().__init__(available_letters, difficulty)
        # Inicializamos los escalares de estado que determinan la heurística de progreso de la sesión activa
        self.streak: int = 0
        # Persistimos el valor máximo del multiplicador temporal para la generación del reporte analítico post-sesión
        self.peak_streak: int = 0          
        self.time_limit: int = 60

    def start_game(self, difficulty: str | None = None) -> str:
        """
        Ejecuta la secuencia de arranque de la máquina de estados, purgando los vectores de sesión residuales y estableciendo el tiempo cero absoluto.
        Retorna el primer objetivo escalar para la inicialización del renderizado.
        """
        super().start_game(difficulty)
        self.streak = 0
        self.peak_streak = 0
        return self.target_letter

    def check_time(self) -> tuple[bool, str]:
        """
        Intercepción del bucle de eventos principal invocada en cada cuadro para garantizar la terminación determinista de la sesión basada en el delta temporal absoluto.
        Retorna una tupla binaria de estado y la cadena formateada de telemetría final.
        """
        if not self.game_active:
            return False, ""
        if self.time_left <= 0:
            self.game_active = False
            return True, f"¡Tiempo! Puntuación: {self.score} pts | Palabras: {self._words_completed}"
        return False, ""

    def check_prediction(self, predicted_letter: str) -> tuple[bool, str]:
        """
        Compara la salida del tensor de clasificación neuronal contra el estado léxico esperado actual.
        Retorna la confirmación de la transacción lógica y el buffer de actualización para la interfaz de usuario.
        """
        if not self.game_active:
            return False, "Juego terminado"

        # Imponemos una barrera de validación temporal previa a la evaluación heurística para evitar vulnerabilidades de latencia en la condición de victoria
        expired, msg = self.check_time()
        if expired:
            return False, msg

        if predicted_letter.upper() == self.target_letter:
            self.streak += 1
            if self.streak > self.peak_streak:
                # Registramos el desplazamiento positivo en la asimetría de rendimiento del usuario
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
        raw = get_words_by_difficulty(self.difficulty)
        # Filtramos la matriz de palabras limitándola estrictamente a la intersección con el subconjunto de clases soportadas por el modelo local
        self._word_pool = [
            w for w in raw
            if all(ch in self.available_letters for ch in w.upper())
        ]
        if not self._word_pool:
            # Implementamos una estrategia de degradación elegante (fallback) para garantizar la continuidad del flujo de ejecución ante un corpus estructuralmente insuficiente
            self._word_pool = self.available_letters[:10] or ["A"]

class SpellingGame(BaseGame):
    """
    Arquitectura de evaluación guiada por retroalimentación acústica.
    El módulo orquesta un pipeline de dictado mediante un subsistema Text-To-Speech (TTS), requiriendo validación secuencial del árbol léxico por parte del motor de visión por computadora.
    Mantiene un registro continuo de la desviación del error (bonus management) sin aplicar decaimiento destructivo a la puntuación base.
    """

    MODE = GameMode.SPELLING

    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        super().__init__(available_letters, difficulty)
        self.time_limit: int = 120          

        # Estado específico de SpellingGame
        self._errors_in_word: int = 0      
        self._total_errors: int = 0        
        # Estructuramos un buffer FIFO asíncrono para delegar el consumo de cargas útiles de audio al hilo principal del sistema operativo
        self.pending_voice: list[str] = []

    def start_game(self, difficulty: str | None = None) -> str:
        """
        Desencadena la inicialización de la matriz de contexto, poblando el buffer de audio inicial para la orquestación del evento de apertura del sistema TTS.
        """
        super().start_game(difficulty)
        self._total_errors = 0
        # Inyectamos el flujo de configuración acústica en el buffer de salida
        self.pending_voice = [
            f"Juego de deletreo.",
            f"Forma la palabra: {self.target_word}",
            f"Primera letra: {self.target_letter}",
        ]
        return self.target_word

    def check_time(self) -> tuple[bool, str]:
        if not self.game_active:
            return False, ""
        if self.time_left <= 0:
            self.game_active = False
            msg = (
                f"¡Tiempo! "
                f"Palabras: {self._words_completed} | "
                f"Puntos: {self.score}"
            )
            self.pending_voice = [
                f"Tiempo terminado. Completaste {self._words_completed} palabras."
            ]
            return True, msg
        return False, ""

    def check_prediction(self, predicted_letter: str) -> tuple[bool, str]:
        # Purgamos la cola de eventos acústicos de la iteración previa
        self.pending_voice = []

        if not self.game_active:
            return False, "Juego terminado"

        expired, msg = self.check_time()
        if expired:
            return False, msg

        if predicted_letter.upper() == self.target_letter:
            # Resolución exitosa de la capa de inferencia
            word_done = self._advance_letter()

            if word_done:
                # Condicionamos la inyección del escalar de recompensa máximo a una tasa de error de cero varianza en la iteración actual
                if self._errors_in_word == 0:
                    self.score += 20
                    bonus_msg = "¡Perfecto! Sin errores."
                else:
                    self.score += 10
                    bonus_msg = "¡Palabra completa!"

                self._words_completed += 1
                self._pick_new_word()

                self.pending_voice = [
                    bonus_msg,
                    f"Siguiente palabra: {self.target_word}",
                    f"Primera letra: {self.target_letter}",
                ]
                return True, f"{bonus_msg} | Pts:{self.score} | Sig: {self.target_word}"
            else:
                self.pending_voice = [f"Correcto. {self.target_letter}"]
                progress = self._build_progress_display()
                return True, f"{progress} | Pts:{self.score} | {self.time_left}s"
        else:
            # Divergencia detectada entre la predicción y el estado objetivo
            self._errors_in_word += 1
            self._total_errors += 1
            self.pending_voice = [f"Intenta de nuevo. Letra: {self.target_letter}"]
            progress = self._build_progress_display()
            return False, f"Error. {progress} | {self.time_left}s"

    @property
    def total_errors(self) -> int:
        return self._total_errors

    def _build_word_pool(self):
        raw = get_spelling_words_by_difficulty(self.difficulty)
        self._word_pool = [
            w for w in raw
            if all(ch in self.available_letters for ch in w.upper())
        ]
        if not self._word_pool:
            self._word_pool = self.available_letters[:5] or ["SOL"]

    def _pick_new_word(self):
        super()._pick_new_word()
        self._errors_in_word = 0

# Preservamos el descriptor histórico SignGame delegando su instanciación dinámicamente hacia la estructura actualizada SpeedGame para garantizar la compatibilidad hacia atrás.
SignGame = SpeedGame