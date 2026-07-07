import random
import time
from enum import Enum, auto

from src.utils import app_logger
from src.modules.word_bank import (
    get_words_by_difficulty,
    get_spelling_words_by_difficulty,
)

# Definimos una estructura enumerada de tipado estricto para gestionar el enrutamiento contextual del ciclo de vida en la capa de orquestación principal
class GameMode(Enum):
    NONE = auto()
    SPEED = auto()      
    SPELLING = auto()   

class SpeedGame:
    # Vinculamos la clase al identificador de enrutamiento estático correspondiente
    MODE = GameMode.SPEED

    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        # Estandarizamos el espacio de clases viables forzando una codificación en mayúsculas para homogeneizar el pipeline de evaluación
        self.available_letters = [l.upper() for l in available_letters]
        self.difficulty = difficulty

        # Inicializamos los escalares de estado que determinan la heurística de progreso de la sesión activa
        self.score: int = 0
        self.streak: int = 0
        # Persistimos el valor máximo del multiplicador temporal para la generación del reporte analítico post-sesión
        self.peak_streak: int = 0          
        self.game_active: bool = False
        self.start_time: float = 0.0
        self.time_limit: int = 60

        # Mantenemos las referencias a los punteros de iteración del corpus de caracteres y strings
        self.target_word: str = ""
        self.target_letter: str = ""
        self._letter_index: int = 0
        self._words_completed: int = 0
        self._word_pool: list[str] = []

    def start_game(self, difficulty: str | None = None) -> str:
        """
        Ejecuta la secuencia de arranque de la máquina de estados, purgando los vectores de sesión residuales y estableciendo el tiempo cero absoluto.
        Retorna el primer objetivo escalar para la inicialización del renderizado.
        """
        if difficulty:
            self.difficulty = difficulty

        self.score = 0
        self.streak = 0
        self.peak_streak = 0
        self._words_completed = 0
        self.game_active = True
        self.start_time = time.time()

        self._build_word_pool()
        self._pick_new_word()
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

    def _pick_new_word(self):
        self.target_word = random.choice(self._word_pool).upper()
        self._letter_index = 0
        self.target_letter = self.target_word[0]

    def _advance_letter(self) -> bool:
        self._letter_index += 1
        if self._letter_index >= len(self.target_word):
            return True
        self.target_letter = self.target_word[self._letter_index]
        return False

    def _build_progress_display(self) -> str:
        """
        Ensamblamos dinámicamente la representación en cadena del buffer de progreso espacial, inyectando delimitadores de formato para resaltar la posición actual del puntero léxico.
        """
        chars = []
        for i, ch in enumerate(self.target_word):
            if i < self._letter_index:
                chars.append(ch)           # Segmento de inferencia validado positivamente
            elif i == self._letter_index:
                chars.append(f"[{ch}]")    # Foco activo del subsistema de evaluación
            else:
                chars.append("_")          # Segmento de inferencia pendiente de evaluación
        return " ".join(chars)


class SpellingGame:
    """
    Arquitectura de evaluación guiada por retroalimentación acústica.
    El módulo orquesta un pipeline de dictado mediante un subsistema Text-To-Speech (TTS), requiriendo validación secuencial del árbol léxico por parte del motor de visión por computadora.
    Mantiene un registro continuo de la desviación del error (bonus management) sin aplicar decaimiento destructivo a la puntuación base.
    """

    MODE = GameMode.SPELLING

    def __init__(self, available_letters: list[str], difficulty: str = "facil"):
        self.available_letters = [l.upper() for l in available_letters]
        self.difficulty = difficulty

        # Variables de estado globales para el control del bucle lógico y el delta de temporización de la sesión
        self.score: int = 0
        self.game_active: bool = False
        self.start_time: float = 0.0
        self.time_limit: int = 120          

        # Controladores de la ventana deslizante para el seguimiento de la convergencia de la cadena de caracteres
        self.target_word: str = ""
        self.target_letter: str = ""
        self._letter_index: int = 0
        self._words_completed: int = 0
        self._errors_in_word: int = 0      
        self._total_errors: int = 0        
        self._word_pool: list[str] = []

        # Estructuramos un buffer FIFO asíncrono para delegar el consumo de cargas útiles de audio al hilo principal del sistema operativo
        self.pending_voice: list[str] = []

    def start_game(self, difficulty: str | None = None) -> str:
        """
        Desencadena la inicialización de la matriz de contexto, poblando el buffer de audio inicial para la orquestación del evento de apertura del sistema TTS.
        """
        if difficulty:
            self.difficulty = difficulty

        self.score = 0
        self._words_completed = 0
        self._total_errors = 0
        self.game_active = True
        self.start_time = time.time()

        self._build_word_pool()
        self._pick_new_word()

        # Inyectamos el flujo de configuración acústica en el buffer de salida
        self.pending_voice = [
            f"Juego de deletreo.",
            f"Forma la palabra: {self.target_word}",
            f"Primera letra: {self.target_letter}",
        ]
        return self.target_word

    def check_time(self) -> tuple[bool, str]:
        """
        Poller de interrupción temporal diseñado para desencadenar el volcado final de estados si el umbral del hardware de reloj de la máquina excede la tolerancia definida.
        """
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
        """
        Compara la salida probabilística del modelo contra el nodo léxico activo. 
        Actualiza el estado asíncrono del sistema y repuebla el buffer de síntesis de voz (pending_voice) basándose en la tasa de falsos positivos detectada.
        """
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
        self.target_word = random.choice(self._word_pool).upper()
        self._letter_index = 0
        self.target_letter = self.target_word[0]
        self._errors_in_word = 0

    def _advance_letter(self) -> bool:
        self._letter_index += 1
        if self._letter_index >= len(self.target_word):
            return True
        self.target_letter = self.target_word[self._letter_index]
        return False

    def _build_progress_display(self) -> str:
        """
        Replicamos la lógica de renderizado matricial de SpeedGame, inyectando marcadores visuales para el seguimiento unívoco de la secuencia de validación actual.
        """
        chars = []
        for i, ch in enumerate(self.target_word):
            if i < self._letter_index:
                chars.append(ch)
            elif i == self._letter_index:
                chars.append(f"[{ch}]")
            else:
                chars.append("_")
        return " ".join(chars)


# Preservamos el descriptor histórico SignGame delegando su instanciación dinámicamente hacia la estructura actualizada SpeedGame para garantizar la compatibilidad hacia atrás.
SignGame = SpeedGame