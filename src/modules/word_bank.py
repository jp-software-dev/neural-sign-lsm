# Definimos un diccionario estático en memoria para mapear la clasificación de complejidad algorítmica hacia vectores contiguos de cadenas de caracteres para el motor de velocidad
WORD_BANK = {
    "facil": [
        "SOL", "MAR", "PAN", "LUZ", "PAZ", "FIN", "RED", "ROL",
        "OJO", "ALA", "PEZ", "GAS", "SER", "VER", "DAR", "HAZ",
        "IRE", "FUE", "OCA", "UVA", "ERA", "OSO", "EJE", "ARO",
    ],
    "medio": [
        "CASA", "MANO", "AMOR", "VIDA", "AGUA", "MASA", "CARA",
        "PALO", "SOPA", "MESA", "HORA", "LAGO", "ROCA", "BOCA",
        "LOCO", "POCO", "RICO", "DURO", "FRIO", "GRIS", "POLO",
        "PUMA", "ROSA", "TELA", "MAPA", "CIMA", "LOMA", "REMO",
    ],
    "dificil": [
        "BRAVO", "CAMPO", "DELTA", "FICHA", "GLOBO", "HUEVO",
        "ICONO", "JUEGO", "LIMON", "MUNDO", "OPERA", "PLUMA",
        "RUMBO", "SALVO", "TECHO", "VAPOR", "SABOR", "PLATO",
        "GRAMO", "FRENO", "TURNO", "BANCO", "COPAL", "ADOBE",
    ],
}

# Estructuramos un grafo tipado de corpus léxico segregado por niveles de entropía para alimentar el pipeline de validación secuencial del motor TTS
SPELLING_WORDS: dict[str, list[str]] = {
    "facil": [
        "SOL", "MAR", "PAN", "OJO", "ALA", "UVA", "OSO",
        "ERA", "IRE", "PAZ", "LUZ", "FIN", "SER", "DAR",
    ],
    "medio": [
        "AMOR", "VIDA", "CASA", "MANO", "LAGO", "ROCA",
        "MESA", "SOPA", "CARA", "PALO", "HORA", "BOCA",
    ],
    "dificil": [
        "MUNDO", "CAMPO", "BRAVO", "JUEGO", "LIMON",
        "PLUMA", "VAPOR", "TECHO", "RUMBO", "SALVO",
    ],
}

# Aplicamos una transformación de aplanamiento iterativo sobre las matrices multidimensionales del corpus para generar un espacio de búsqueda unidimensional global
ALL_WORDS = [w for words in WORD_BANK.values() for w in words]

# Compilamos estáticamente una lista plana unificada del corpus de deletreo forzando un tipado estricto para garantizar la integridad referencial
ALL_SPELLING_WORDS: list[str] = [w for words in SPELLING_WORDS.values() for w in words]


def get_words_by_difficulty(difficulty: str) -> list[str]:
    # Interroga la estructura de datos léxica inyectando una estrategia de degradación (fallback) hacia el tensor global si el apuntador de la clave falla
    return WORD_BANK.get(difficulty.lower(), ALL_WORDS)


def get_spelling_words_by_difficulty(difficulty: str) -> list[str]:
    # Expone una interfaz de consulta determinista para extraer particiones de deletreo delegando la normalización de la cadena al compilador
    return SPELLING_WORDS.get(difficulty.lower(), ALL_SPELLING_WORDS)


def get_all_words() -> list[str]:
    # Retorna un puntero de solo lectura a la colección maestra de entidades de prueba por velocidad
    return ALL_WORDS


def get_all_spelling_words() -> list[str]:
    # Transmite la referencia de memoria de la matriz plana de evaluación fonética hacia las subrutinas consumidoras
    return ALL_SPELLING_WORDS