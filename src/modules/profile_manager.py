"""Gestiona la carga y guardado de los puntajes más altos."""
import json
import os
from src.config.settings import PROJECT_ROOT

HIGH_SCORES_PATH = os.path.join(PROJECT_ROOT, "data", "high_scores.json")
MAX_SCORES = 3

def load_high_scores() -> list[dict]:
    """Carga la lista de mejores puntajes desde un archivo JSON."""
    if not os.path.exists(HIGH_SCORES_PATH):
        return []
    try:
        with open(HIGH_SCORES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_high_scores(scores: list[dict]):
    """Guarda la lista de mejores puntajes en un archivo JSON."""
    try:
        os.makedirs(os.path.dirname(HIGH_SCORES_PATH), exist_ok=True)
        with open(HIGH_SCORES_PATH, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=4)
    except IOError as e:
        print(f"Error guardando los puntajes: {e}")

def is_high_score(score: int, scores: list[dict]) -> bool:
    """Verifica si un puntaje es lo suficientemente alto para entrar en el Top 3."""
    return len(scores) < MAX_SCORES or score > scores[-1].get('score', 0)

def add_high_score(name: str, score: int, scores: list[dict]) -> list[dict]:
    """Añade un nuevo puntaje a la lista, la ordena y la mantiene en el Top 3."""
    scores.append({'name': name.strip(), 'score': score})
    scores.sort(key=lambda x: x.get('score', 0), reverse=True)
    return scores[:MAX_SCORES]