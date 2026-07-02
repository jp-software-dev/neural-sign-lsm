import pyttsx3
import threading
import queue
import time
from src.utils import app_logger

class VoiceAssistant:
    def __init__(self):
        # Instanciamos una estructura FIFO en el bloque heap orientada a hilos seguros para aislar el hardware de audio del pipeline de inferencia
        self.speech_queue = queue.Queue()
        
        # Inicializamos el hilo consumidor asignando un modo daemon para autorizar la destruccion de memoria cuando finaliza el hilo primario
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        # Transmitimos la senal syscall de inicializacion del hilo subyacente hacia el scheduler del sistema operativo local
        self._worker_thread.start()

    def _speech_worker(self):
        try:
            # Instanciamos el binding nativo COM en entornos Windows o espeak en entornos Linux estrictamente dentro del puntero del hilo secundario
            self.engine = pyttsx3.init()
            
            # Parametrizamos el atributo interno rate del motor sintético para reducir iteraciones de palabras por minuto limitando la distorsion 
            self.engine.setProperty('rate', 150)
            
            # Saturamos el multiplicador del bus del puerto maestro del emulador estableciendo la emision del decodificador con amplitud pico nativa
            self.engine.setProperty('volume', 1.0)

            # Extraemos un arreglo referencial de los metadatos de los modelos vocales registrados localmente en el registro de hardware
            voices = self.engine.getProperty('voices')
            
            # Iteramos secuencialmente los metadatos analizando la firma lexica del identificador SAPI para detectar fonetica de sintesis en espanol
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower() or 'mexico' in voice.name.lower():
                    # Acoplamos el identificador del hardware al motor TTS sobrescribiendo el perfil por defecto del SO para forzar la emision latina
                    self.engine.setProperty('voice', voice.id)
                    break
                    
        except Exception as e:
            # Capturamos excepciones COM derivadas del motor de sonido para inyectarlas directamente al pool de telemetria omitiendo interrupcion
            app_logger.error(f"Error inicializando VoiceAssistant en el hilo: {str(e)}")
            return

        # Anclamos la maquina de estados del trabajador en un loop residente eterno de evaluacion ininterrumpida de mensajes inter-procesos
        while True:
            try:
                # Extraemos el puntero superficial del nodo en cola invocando un bloqueo cooperativo del contexto hasta localizar datos en memoria 
                text = self.speech_queue.get()
                
                # Evaluamos de forma estricta si se aprovisiono un nulo referencial para destruir el stack y desencadenar una salida limpia del worker
                if text is None:
                    break
                    
                # Invocamos la transformacion computacional PCM instruyendo al motor sintetizador a procesar la directiva en un stream continuo
                self.engine.say(text)
                # Aplicamos un bloqueo sincronico sobre la maquina de estados del hilo secundario hasta purgar la totalidad del buffer en la salida DAC
                self.engine.runAndWait()
                
                # Emitimos una senal al controlador semaforo de la estructura Queue notificando la eliminacion y resolucion final de la tarea
                self.speech_queue.task_done()
                
            except Exception as e:
                # Interceptamos desbordamientos o colisiones I/O despachando trazas formales a los logs para prevenir volcados de error en std_out
                app_logger.error(f"Error del motor TTS durante la reproduccion: {str(e)}")

    def speak(self, text):
        # Medimos el volumen de carga actual del sistema FIFO limitando estrictamente el buffer para prevenir desincronizacion acustica en tiempo real
        if self.speech_queue.qsize() > 2:
            # Adquirimos posesion temporal exclusiva del mutex transaccional impidiendo operaciones de lectura del hilo secundario de sintesis
            with self.speech_queue.mutex:
                # Purgamos y limpiamos por completo la cola deque subyacente forzando descartes para mantener unicamente la inferencia mas critica y reciente
                self.speech_queue.queue.clear()
                
        # Inyectamos el puntero a la cadena de caracteres hacia la pila dinamica delegando la traduccion fonetica al daemon asincrono
        self.speech_queue.put(text)

if __name__ == '__main__':
    # Construimos el objeto proxy principal activando colateralmente la inicializacion asincrona del hardware de sintesis
    bot = VoiceAssistant()
    # Enviamos una carga util emulada al bus de eventos para corroborar el mapeo del motor auditivo al subsistema I/O operativo
    bot.speak("Iniciando sistema de traduccion de Lengua de Senas Mexicana.")
    # Exigimos un letargo preventivo sobre el main thread cediendo procesamiento condicional al daemon auxiliar antes de invocar su destruccion
    time.sleep(3)