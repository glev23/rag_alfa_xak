"""Конфигурация для RAG системы."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RAG_COLLECTION_DIR = BASE_DIR / os.getenv("RAG_COLLECTION_DIR", "data/rag_collection")
SUBMISSION_DIR = BASE_DIR / os.getenv("SUBMISSION_DIR", "submissions")

# Создание директорий если не существуют
RAG_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Qdrant настройки
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6883"))

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Embedding модели
EMBEDDING_DIMENSION_FRIDA = int(os.getenv("EMBEDDING_DIMENSION_FRIDA", "1536"))

# RAG параметры
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Файлы данных
WEBSITES_FILE = DATA_DIR / "websites_cleaned.csv"
QUESTIONS_FILE = DATA_DIR / "questions_clean.csv"

# Коллекции Qdrant
COLLECTION_FRIDA = "bank_site_frida"
COLLECTION_FRIDA_512 = "bank_site_frida_512"

# Модели
MODEL_FRIDA = "ai-forever/FRIDA"

# Отладка
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

