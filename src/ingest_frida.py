"""
Скрипт для создания эмбеддингов с помощью FRIDA (512 токенов) и загрузки в Qdrant.
Модель: ai-forever/FRIDA
Размерность: 1536
Использует chunks_frida.json (созданный prepare_chunks_frida.py)
"""

import json
import os
import time
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import (
    COLLECTION_FRIDA,
    EMBEDDING_DIMENSION_FRIDA,
    HF_TOKEN,
    MODEL_FRIDA,
    QDRANT_HOST,
    QDRANT_PORT,
    RAG_COLLECTION_DIR,
)


def load_chunks(input_file: str):
    """Загружает чанки из JSON файла."""
    chunks_file = RAG_COLLECTION_DIR / input_file
    
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Файл с чанками не найден: {chunks_file}\n"
            "Укажите правильный путь к файлу с чанками"
        )
    
    print(f"📖 Загрузка чанков из {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✅ Загружено {len(chunks)} чанков")
    return chunks


def create_qdrant_collection(client: QdrantClient, embedding_dim: int, collection_name: str):
    """Создает коллекцию в Qdrant или пересоздает если существует."""
    
    print(f"\n🗄️  Подготовка коллекции '{collection_name}' в Qdrant...")
    print(f"   Размерность векторов: {embedding_dim}")
    
    # Проверяем существование коллекции
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if collection_name in collection_names:
        print(f"⚠️  Коллекция '{collection_name}' уже существует. Удаляем...")
        try:
            client.delete_collection(collection_name)
            print(f"✅ Старая коллекция удалена")
        except Exception as e:
            print(f"⚠️  Ошибка при удалении коллекции: {e}")
    
    # Создаем новую коллекцию
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    
    print(f"✅ Коллекция '{collection_name}' создана с размерностью {embedding_dim}")


def embed_and_upload(chunks: list[dict], collection_name: str):
    """
    Создает эмбеддинги для чанков и загружает в Qdrant.
    
    Args:
        chunks: Список чанков с метаданными
        collection_name: Название коллекции в Qdrant
    """
    
    print(f"\n🤖 Загрузка модели {MODEL_FRIDA}...")
    
    # Определяем устройство
    if not torch.cuda.is_available():
        device = "cpu"
        print(f"⚠️  CUDA недоступна, используется CPU")
    else:
        # Проверяем переменную окружения CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            print(f"📌 CUDA_VISIBLE_DEVICES={cuda_visible}")
            # Если установлен CUDA_VISIBLE_DEVICES, используем cuda:0 (первая видимая карта)
            device = "cuda:0"
        else:
            # По умолчанию используем первую доступную карту
            device = "cuda:0"
            print(f"💡 Используется GPU 0 (по умолчанию)")
    
    print(f"Устройство: {device}")
    
    # Показываем информацию о доступных GPU
    if torch.cuda.is_available():
        print(f"📊 Доступно GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Загрузка модели FRIDA
    model = SentenceTransformer(
        MODEL_FRIDA,
        device=device,
        trust_remote_code=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    print(f"✅ Модель загружена")
    
    # Проверяем размерность модели
    test_embedding = model.encode(["test"], convert_to_numpy=True)[0]
    actual_dim = len(test_embedding)
    print(f"📏 Размерность эмбеддинга модели: {actual_dim}")
    
    if actual_dim != EMBEDDING_DIMENSION_FRIDA:
        print(f"⚠️  ВНИМАНИЕ: Ожидалась размерность {EMBEDDING_DIMENSION_FRIDA}, но модель возвращает {actual_dim}")
        print(f"   Используем фактическую размерность модели: {actual_dim}")
    
    # Используем фактическую размерность модели
    embedding_dim = actual_dim
    
    # Подключение к Qdrant
    print(f"\n🔗 Подключение к Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Создаем коллекцию (пересоздаст если существует)
    create_qdrant_collection(client, embedding_dim, collection_name)
    
    print(f"\n🚀 Начинаем создание эмбеддингов и загрузку в Qdrant...")
    print(f"Обработка по одному тексту (для экономии памяти на длинных текстах)")
    print(f"Используем префикс 'search_document: ' для документов (FRIDA)")
    
    start_time = time.time()
    
    # Обрабатываем по одному тексту за раз для экономии памяти
    points_batch = []
    batch_size_upload = 100  # Размер батча для загрузки в Qdrant
    
    for chunk in tqdm(chunks, desc="Обработка чанков"):
        text = chunk["text"]
        
        # FRIDA использует префикс "search_document: " для документов
        text_with_prefix = f"search_document: {text}"
        
        try:
            # Создаем эмбеддинг для одного текста
            embedding = model.encode(
                [text_with_prefix],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
            
            # Проверяем размерность эмбеддинга
            if len(embedding) != embedding_dim:
                print(f"\n❌ Ошибка размерности на chunk_id={chunk['chunk_id']}: ожидалось {embedding_dim}, получено {len(embedding)}")
                continue
            
            # Подготавливаем точку для Qdrant
            point = PointStruct(
                id=chunk["chunk_id"],
                vector=embedding.tolist(),
                payload={
                    "web_id": chunk["web_id"],
                    "url": chunk["url"],
                    "kind": chunk["kind"],
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "text": chunk["text"],
                },
            )
            points_batch.append(point)
            
            # Загружаем в Qdrant батчами для эффективности
            if len(points_batch) >= batch_size_upload:
                client.upsert(collection_name=collection_name, points=points_batch)
                points_batch = []
            
            # Очищаем кэш GPU после каждого чанка
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except torch.cuda.OutOfMemoryError:
            print(f"\n❌ Out of Memory на chunk_id={chunk['chunk_id']}, web_id={chunk['web_id']}")
            print(f"   Длина текста: {len(text)} символов")
            # Очищаем память и пропускаем этот чанк
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
    
    # Загружаем оставшиеся точки
    if points_batch:
        client.upsert(collection_name=collection_name, points=points_batch)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Готово!")
    print(f"   Обработано: {len(chunks)} чанков")
    print(f"   Время: {elapsed_time:.2f} секунд")
    print(f"   Скорость: {len(chunks) / elapsed_time:.2f} чанков/сек")
    
    # Проверяем количество точек в коллекции
    collection_info = client.get_collection(collection_name)
    print(f"\n📊 Информация о коллекции:")
    print(f"   Название: {collection_name}")
    print(f"   Точек: {collection_info.points_count}")
    print(f"   Размерность: {embedding_dim}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Создание эмбеддингов FRIDA и загрузка в Qdrant")
    parser.add_argument("--input", type=str, required=True, help="Имя входного файла с чанками (например: chunks_frida_500_100.json)")
    parser.add_argument("--collection", type=str, required=True, help="Название коллекции в Qdrant (например: bank_site_frida_500_100)")
    
    args = parser.parse_args()
    
    chunks = load_chunks(args.input)
    embed_and_upload(chunks, args.collection)

