import pyttsx3
import threading
import queue
import time
from src.utils import app_logger

class VoiceAssistant:
    def __init__(self):
        # Cola segura para hilos para desacoplar la síntesis de voz del hilo principal.
        self.speech_queue = queue.Queue()
        # Hilo de trabajo (worker) que consume de la cola. Se ejecuta en segundo plano.
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _speech_worker(self):
        """
        Procesa la cola de síntesis de voz en un hilo separado.
        Esto evita que el motor TTS (pyttsx3) bloquee el hilo principal (ej. el bucle de OpenCV).
        """
        try:
            # Inicializa el motor TTS dentro del hilo para seguridad de concurrencia.
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)

            # Busca y establece una voz en español si está disponible.
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower() or 'mexico' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            app_logger.error(f"Error inicializando VoiceAssistant en el hilo: {str(e)}")
            return

        # Bucle principal del worker: espera y procesa texto de la cola.
        while True:
            try:
                # .get() es bloqueante: espera hasta que haya un ítem en la cola.
                text = self.speech_queue.get()                
                if text is None:
                    break
                
                self.engine.say(text)
                # .runAndWait() bloquea este hilo (el worker), no el principal.
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except Exception as e:
                app_logger.error(f"Error del motor TTS durante la reproduccion: {str(e)}")

    def speak(self, text):
        # Evita la acumulación excesiva de mensajes de voz para mantener la responsividad.
        if self.speech_queue.qsize() > 2:
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        # Añade el texto a la cola para ser procesado por el hilo worker.
        self.speech_queue.put(text)

if __name__ == '__main__':
    # Construimos el objeto proxy principal activando colateralmente la inicializacion asincrona del hardware de sintesis
    bot = VoiceAssistant()
    # Enviamos una carga util emulada al bus de eventos para corroborar el mapeo del motor auditivo al subsistema I/O operativo
    bot.speak("Iniciando sistema de traduccion de Lengua de Senas Mexicana.")
    # Exigimos un letargo preventivo sobre el main thread cediendo procesamiento condicional al daemon auxiliar antes de invocar su destruccion
    time.sleep(3)