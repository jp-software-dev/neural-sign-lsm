"""Define el banco de palabras para los modos de juego, organizado por dificultad."""
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

ALL_WORDS = [w for words in WORD_BANK.values() for w in words]

def get_words_by_difficulty(difficulty: str) -> list[str]:
    """Devuelve una lista de palabras basada en el nivel de dificultad."""
    if difficulty.lower() == "aleatorio":
        return ALL_WORDS
    return WORD_BANK.get(difficulty.lower(), ALL_WORDS)

def get_all_words() -> list[str]:
    """Devuelve una lista con todas las palabras disponibles."""
    return ALL_WORDS