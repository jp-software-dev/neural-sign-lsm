"""
word_bank.py
------------
Banco de palabras para el minijuego de LSM.
Las palabras están limitadas a letras que el modelo puede clasificar (A-Z sin Ñ ni dígrafos).
Se organizan por dificultad creciente según longitud y frecuencia de letras difíciles.
"""

# ==========================================
# Banco de palabras agrupadas por dificultad
# ==========================================

WORD_BANK = {
    "facil": [
        "SOL", "MAR", "PAN", "LUZ", "PAZ", "FIN", "RED", "ROL",
        "OJO", "ALA", "PEZ", "BOX", "GAS", "COR", "SER", "VER",
        "DAR", "IRE", "FUE", "HAZ",
    ],
    "medio": [
        "CASA", "MANO", "AMOR", "VIDA", "AGUA", "MASA", "CARA",
        "PALO", "SOPA", "MESA", "HORA", "LAGO", "ROCA", "BOCA",
        "LOCO", "POCO", "RICO", "DURO", "FRIO", "GRIS",
    ],
    "dificil": [
        "BRAVO", "CAMPO", "DELTA", "FICHA", "GLOBO", "HUEVO",
        "ICONO", "JUEGO", "KARMA", "LIMON", "MUNDO", "NEXUS",
        "OPERA", "PLUMA", "QUESO", "RUMBO", "SALVO", "TECHO",
        "UMBRA", "VAPOR",
    ],
}

# Todas las palabras en una lista plana para selección aleatoria simple
ALL_WORDS = [w for words in WORD_BANK.values() for w in words]


def get_words_by_difficulty(difficulty: str) -> list[str]:
    """
    Retorna la lista de palabras para la dificultad dada.
    Acepta: 'facil', 'medio', 'dificil'.
    Si la dificultad no existe retorna todas las palabras.
    """
    return WORD_BANK.get(difficulty.lower(), ALL_WORDS)


def get_all_words() -> list[str]:
    """Retorna el banco completo de palabras."""
    return ALL_WORDS