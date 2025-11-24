"""
RAG Retrieval с использованием FRIDA эмбеддингов (512 токенов).
Ищет топ-5 релевантных документов для каждого вопроса.
Использует коллекцию bank_site_frida_512
"""

import time
from datetime import datetime

import pandas as pd
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import (
    COLLECTION_FRIDA_512,
    HF_TOKEN,
    MODEL_FRIDA,
    QDRANT_HOST,
    QDRANT_PORT,
    QUESTIONS_FILE,
    SUBMISSION_DIR,
    TOP_K_RESULTS,
)


def load_questions():
    """Загружает вопросы из CSV файла."""
    print(f"📖 Загрузка вопросов из {QUESTIONS_FILE}...")
    df = pd.read_csv(QUESTIONS_FILE)
    print(f"✅ Загружено {len(df)} вопросов")
    return df


def retrieve_documents(
    questions_df: pd.DataFrame,
    collection_name: str,
    batch_size: int = 32,
    top_k: int = TOP_K_RESULTS,
):
    """
    Ищет топ-k релевантных документов для каждого вопроса.
    
    Args:
        questions_df: DataFrame с вопросами (q_id, query)
        collection_name: Название коллекции в Qdrant
        batch_size: Размер батча для обработки
        top_k: Количество возвращаемых документов
        
    Returns:
        DataFrame с результатами (q_id, web_list)
    """
    
    print(f"\n🤖 Загрузка модели {MODEL_FRIDA}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    
    model = SentenceTransformer(
        MODEL_FRIDA,
        device=device,
        trust_remote_code=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    print(f"✅ Модель загружена")
    
    print(f"\n🔗 Подключение к Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Проверяем существование коллекции
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if collection_name not in collection_names:
        raise ValueError(
            f"Коллекция '{collection_name}' не найдена в Qdrant!\n"
            f"Доступные коллекции: {', '.join(collection_names)}"
        )
    
    print(f"✅ Коллекция '{collection_name}' найдена")
    
    results = []
    queries = questions_df["query"].tolist()
    q_ids = questions_df["q_id"].tolist()
    
    print(f"\n🔍 Начинаем поиск документов (top-{top_k})...")
    print(f"Используем префикс 'search_query: ' для вопросов (FRIDA)")
    start_time = time.time()
    
    # Обработка батчами с прогресс-баром по вопросам
    pbar = tqdm(total=len(queries), desc="Обработка вопросов")
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]
        batch_q_ids = q_ids[i : i + batch_size]
        
        # Добавляем префикс для запросов
        batch_queries_with_prefix = [f"search_query: {q}" for q in batch_queries]
        
        # Создаем эмбеддинги для вопросов
        query_embeddings = model.encode(
            batch_queries_with_prefix,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Поиск для каждого вопроса в батче
        for q_id, query_embedding in zip(batch_q_ids, query_embeddings):
            # Поиск в Qdrant (возвращаем больше чанков для последующей агрегации)
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k * 10,  # Увеличиваем для гарантии получения 5 уникальных web_id
            )
            
            # Агрегируем по web_id (берем лучший score для каждого web_id)
            web_id_scores = {}
            for hit in search_results:
                web_id = hit.payload["web_id"]
                score = hit.score
                
                if web_id not in web_id_scores or score > web_id_scores[web_id]:
                    web_id_scores[web_id] = score
            
            # Сортируем по score и берем топ-k
            top_web_ids = sorted(web_id_scores.items(), key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            top_web_ids = [web_id for web_id, _ in top_web_ids]
            
            # Если не хватило уникальных web_id, дополняем
            if len(top_web_ids) < top_k:
                all_web_ids = [hit.payload["web_id"] for hit in search_results]
                for web_id in all_web_ids:
                    if web_id not in top_web_ids:
                        top_web_ids.append(web_id)
                        if len(top_web_ids) >= top_k:
                            break
            
            # Берем ровно top_k элементов
            top_web_ids = top_web_ids[:top_k]
            
            results.append({"q_id": q_id, "web_list": top_web_ids})
            
            # Обновляем прогресс-бар
            pbar.update(1)
    
    pbar.close()
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Поиск завершен!")
    print(f"   Обработано: {len(queries)} вопросов")
    print(f"   Время: {elapsed_time:.2f} секунд")
    print(f"   Скорость: {len(queries) / elapsed_time:.2f} вопросов/сек")
    
    return pd.DataFrame(results)


def save_submission(results_df: pd.DataFrame, model_name: str = "frida_512"):
    """Сохраняет результаты в формате submission."""
    
    # Форматируем web_list как строку со списком
    results_df["web_list"] = results_df["web_list"].apply(lambda x: str(x))
    
    # Генерируем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{model_name}_{timestamp}.csv"
    output_path = SUBMISSION_DIR / filename
    
    print(f"\n💾 Сохранение результатов в {output_path}...")
    results_df.to_csv(output_path, index=False)
    
    print(f"✅ Submission файл сохранен!")
    print(f"   Файл: {output_path}")
    print(f"   Вопросов: {len(results_df)}")
    
    # Показываем примеры
    print(f"\n📋 Примеры результатов:")
    for i, row in results_df.head(5).iterrows():
        print(f"   q_id={row['q_id']}: {row['web_list']}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Retrieval с использованием FRIDA")
    parser.add_argument("--collection", type=str, required=True, help="Название коллекции в Qdrant (например: bank_site_frida_500_100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча для обработки")
    parser.add_argument("--top-k", type=int, default=TOP_K_RESULTS, help="Количество возвращаемых документов")
    parser.add_argument("--model-name", type=str, default="frida_512", help="Имя модели для названия submission файла")
    
    args = parser.parse_args()
    
    # Загружаем вопросы
    questions_df = load_questions()
    
    # Выполняем поиск
    results_df = retrieve_documents(
        questions_df, 
        collection_name=args.collection,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
    
    # Сохраняем submission
    save_submission(results_df, model_name=args.model_name)

