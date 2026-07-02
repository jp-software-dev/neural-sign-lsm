import logging
import os 
from src.config.settings import LOGS_DIR

def setup_logger():
    # Invocamos la creacion del arbol de directorios de manera recursiva asegurando que el path de destino exista antes de inicializar el stream de entrada y salida
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR) 

    # Construimos la ruta absoluta del archivo de volcado concatenando el directorio configurado con el nombre estandarizado del log para el sistema operativo
    log_file_path = os.path.join(LOGS_DIR, "app_errors.log")

    # Configuramos el root logger del modulo logging asignando un FileHandler con nivel ERROR para filtrar mensajes de severidad inferior y formateando el timestamp
    logging.basicConfig(
        filename=log_file_path,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Instanciamos y retornamos un logger nombrado especificamente para aislar la telemetria de nuestra aplicacion del resto de dependencias de terceros
    return logging.getLogger("NeuralSignLogger")

# Exponemos el singleton del logger al espacio de nombres global para centralizar la emision de trazas de error desde cualquier modulo de la arquitectura
app_logger = setup_logger()