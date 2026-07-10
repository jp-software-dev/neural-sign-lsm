# Definimos un diccionario estático en memoria para mapear la clasificación de complejidad algorítmica hacia vectores contiguos de cadenas de caracteres
WORD_BANK = {
    "facil": [
        "SOL", "MAR", "PAN", "LUZ", "PAZ", "FIN", "RED", "ROL", "OJO", "ALA", 
        "PEZ", "GAS", "SER", "VER", "DAR", "HAZ", "IRE", "FUE", "OCA", "UVA", 
        "ERA", "OSO", "EJE", "ARO", "DIA", "MES", "AÑO", "SUR", "NOR", "EST",
        "REY", "LEY", "VOZ", "TOS", "PIE", "SAL", "MIEL", "FLOR", "TREN", "CRUZ",
        "CIEN", "MIL", "DOS", "TRES", "SEIS", "DIEZ", "ONCE", "DOCE", "CIELO"
    ],
    "medio": [
        "CASA", "MANO", "AMOR", "VIDA", "AGUA", "MASA", "CARA", "PALO", "SOPA", 
        "MESA", "HORA", "LAGO", "ROCA", "BOCA", "LOCO", "POCO", "RICO", "DURO", 
        "FRIO", "GRIS", "POLO", "PUMA", "ROSA", "TELA", "MAPA", "CIMA", "LOMA", 
        "REMO", "ARBOL", "PERRO", "GATO", "RATON", "TIGRE", "LEON", "OSO", "LOBO",
        "ZORRO", "MONO", "VACA", "TORO", "PATO", "PAVO", "GALLO", "PEZ", "RANA",
        "PLATO", "VASO", "TAZA", "COPA", "OLLA", "SARTEN", "CUCHARA", "TENEDOR",
        "CUCHILLO", "CAMA", "SILLA", "SOFA", "PUERTA", "VENTANA", "PARED", "TECHO"
    ],
    "dificil": [
        "BRAVO", "CAMPO", "DELTA", "FICHA", "GLOBO", "HUEVO", "ICONO", "JUEGO", 
        "LIMON", "MUNDO", "OPERA", "PLUMA", "RUMBO", "SALVO", "TECHO", "VAPOR", 
        "SABOR", "PLATO", "GRAMO", "FRENO", "TURNO", "BANCO", "COPAL", "ADOBE",
        "COMPUTADORA", "INTERNET", "PROGRAMA", "SISTEMA", "TECLADO", "PANTALLA", 
        "MEMORIA", "ARCHIVO", "CARPETA", "IMAGEN", "VIDEO", "MUSICA", "SONIDO",
        "UNIVERSIDAD", "ESCUELA", "COLEGIO", "MAESTRO", "ALUMNO", "CLASE", "CURSO",
        "EXAMEN", "PRUEBA", "TAREA", "TRABAJO", "PROYECTO", "INVESTIGACION", "CIENCIA",
        "HISTORIA", "GEOGRAFIA", "MATEMATICAS", "LENGUAJE", "ARTE", "DEPORTE", "FUTBOL",
        "BALONCESTO", "NATACION", "ATLETISMO", "GIMNASIA", "TENIS", "VOLEIBOL", "BEISBOL",
        "HOSPITAL", "MEDICO", "ENFERMERA", "PACIENTE", "MEDICINA", "SALUD", "ENFERMEDAD"
    ],
}

# Estructuramos un grafo tipado de corpus léxico idéntico para el motor de deletreo (TTS)
SPELLING_WORDS: dict[str, list[str]] = {
    "facil": WORD_BANK["facil"].copy(),
    "medio": WORD_BANK["medio"].copy(),
    "dificil": WORD_BANK["dificil"].copy(),
}

# Aplicamos una transformación de aplanamiento iterativo sobre las matrices multidimensionales del corpus para generar un espacio de búsqueda unidimensional global
ALL_WORDS = [w for words in WORD_BANK.values() for w in words]
ALL_SPELLING_WORDS: list[str] = [w for words in SPELLING_WORDS.values() for w in words]

def get_words_by_difficulty(difficulty: str) -> list[str]:
    # Interroga la estructura de datos léxica permitiendo extracción aleatoria total
    if difficulty.lower() == "aleatorio":
        return ALL_WORDS
    return WORD_BANK.get(difficulty.lower(), ALL_WORDS)

def get_spelling_words_by_difficulty(difficulty: str) -> list[str]:
    # Expone una interfaz de consulta determinista para extraer particiones de deletreo
    if difficulty.lower() == "aleatorio":
        return ALL_SPELLING_WORDS
    return SPELLING_WORDS.get(difficulty.lower(), ALL_SPELLING_WORDS)

def get_all_words() -> list[str]:
    return ALL_WORDS

def get_all_spelling_words() -> list[str]:
    return ALL_SPELLING_WORDS