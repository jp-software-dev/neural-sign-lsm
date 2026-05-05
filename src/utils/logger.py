import logging
import os 
from src.config.settings import LOGS_DIR

def setup_logger():
    # Crear la carpeta de logs si no existe
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR) 

    # Ruta exacta dela archivo de errores
    log_file_path = os.path.join(LOGS_DIR, "app_error.log")

# Configuración del formato de los errores
    logging.basicConfig(
        filename=log_file_path,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    return logging.getLogger("NeuralSignLogger")

# Instancia global
app_logger = setup_logger()