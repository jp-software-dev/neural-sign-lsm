"""
Define el asistente de voz que opera de forma asíncrona en un hilo secundario
para no bloquear la interfaz principal.
"""
import pyttsx3
import threading
import queue
import time
from src.utils import app_logger

class VoiceAssistant:
    """Gestiona la síntesis de voz (TTS) usando una cola y un hilo de trabajo."""
    def __init__(self):
        self.speech_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _speech_worker(self):
        """Procesa la cola de texto a voz en un hilo separado para evitar bloqueos."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)

            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower() or 'mexico' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            app_logger.error(f"Error inicializando VoiceAssistant en el hilo: {str(e)}")
            return

        while True:
            try:
                text = self.speech_queue.get()
                if text is None:
                    break

                self.engine.say(text)
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except Exception as e:
                app_logger.error(f"Error del motor TTS durante la reproduccion: {str(e)}")

    def speak(self, text):
        """Añade un texto a la cola para ser pronunciado por el asistente."""
        if self.speech_queue.qsize() > 2:
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        self.speech_queue.put(text)

if __name__ == '__main__':
    bot = VoiceAssistant()
    bot.speak("Iniciando sistema de traduccion de Lengua de Senas Mexicana.")
    time.sleep(3)