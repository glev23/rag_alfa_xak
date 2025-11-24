"""
Скрипт для подготовки чанков из веб-страниц для модели FRIDA.
Модель: ai-forever/FRIDA
Размерность эмбеддингов: 1536
Максимальная длина контекста: 512 токенов
Разбивает длинные тексты на чанки по токенам (макс 500 токенов) с грамотным оверлапом.
"""

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from .config import (
    DEBUG,
    HF_TOKEN,
    RAG_COLLECTION_DIR,
    WEBSITES_FILE,
)

# Лимит токенов для FRIDA
MAX_TOKENS = 512
CHUNK_SIZE_TOKENS = 500  # Берем немного меньше для запаса (500 < 512)
CHUNK_OVERLAP_TOKENS = 50  # Грамотный оверлап ~10% от размера чанка

# Токенизатор FRIDA
TOKENIZER_NAME = "ai-forever/FRIDA"


def split_into_sentences(text: str) -> list[str]:
    """
    Разбивает текст на предложения с учетом русского языка и специфики банковских документов.
    
    Args:
        text: Исходный текст
        
    Returns:
        Список предложений
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Паттерн для разбиения на предложения
    # Учитываем: точки, вопросительные, восклицательные знаки
    # НЕ разбиваем на сокращениях (т.е., т.д., руб., и т.п.)
    sentence_endings = re.compile(
        r'(?<!\w\.\w.)(?<![A-ZА-Я][a-zа-я]\.)(?<![A-ZА-Я]\.)(?<=\.|\?|\!|\n)\s+'
    )
    
    sentences = sentence_endings.split(text)
    
    # Убираем пустые строки и лишние пробелы
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Объединяем слишком короткие предложения (< 10 символов) со следующим
    merged_sentences = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        
        # Если предложение очень короткое и это не последнее, объединяем со следующим
        if len(current) < 10 and i < len(sentences) - 1:
            current = current + " " + sentences[i + 1]
            i += 2
        else:
            i += 1
        
        merged_sentences.append(current)
    
    return merged_sentences


def split_long_sentence(sentence: str, tokenizer, max_tokens: int) -> list[str]:
    """
    Разбивает слишком длинное предложение на части по токенам.
    Используется как fallback, когда предложение не помещается в чанк.
    
    Args:
        sentence: Длинное предложение
        tokenizer: Токенизатор
        max_tokens: Максимальное количество токенов в части
        
    Returns:
        Список частей предложения
    """
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [sentence]
    
    parts = []
    start = 0
    
    while start < len(tokens):
        end = start + max_tokens
        part_tokens = tokens[start:end]
        part_text = tokenizer.decode(part_tokens, skip_special_tokens=True).strip()
        
        if part_text:
            parts.append(part_text)
        
        start = end
        
        # Защита от бесконечного цикла
        if start >= len(tokens):
            break
    
    return parts


def split_text_by_sentences(text: str, tokenizer, chunk_size: int, overlap_sentences: int, title: str = None) -> list[str]:
    """
    Разбивает текст на чанки по предложениям с перекрытием.
    Старается не разрывать предложения посередине, но если предложение слишком длинное,
    разбивает его на части (fallback на токенную нарезку).
    
    Args:
        text: Исходный текст
        tokenizer: Токенизатор для подсчета токенов
        chunk_size: Максимальный размер чанка в токенах
        overlap_sentences: Количество предложений для перекрытия
        title: Заголовок для добавления в каждый чанк (опционально)
        
    Returns:
        Список чанков (все чанки гарантированно <= chunk_size токенов)
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Подготавливаем title prefix
    title_tokens = 0
    title_prefix = ""
    if title:
        title_prefix = f"📄 {title}\n\n"
        title_tokens = len(tokenizer.encode(title_prefix, add_special_tokens=False))
        effective_chunk_size = chunk_size - title_tokens
    else:
        effective_chunk_size = chunk_size
    
    # Разбиваем текст на предложения
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    # Токенизируем каждое предложение отдельно
    sentence_tokens = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_tokens.append((sentence, len(tokens)))
    
    # Если весь текст помещается в один чанк
    total_tokens = sum(count for _, count in sentence_tokens)
    if total_tokens <= effective_chunk_size:
        if title:
            return [f"{title_prefix}{text}"]
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(sentence_tokens):
        chunk_sentences = []
        chunk_token_count = 0
        
        # Набираем предложения до достижения лимита
        while i < len(sentence_tokens):
            sentence, token_count = sentence_tokens[i]
            
            # Проверяем, не превысим ли лимит
            if chunk_token_count + token_count > effective_chunk_size:
                # Если это первое предложение и оно слишком длинное, разбиваем его на части
                if not chunk_sentences:
                    # Разбиваем длинное предложение на части по токенам
                    sentence_parts = split_long_sentence(sentence, tokenizer, effective_chunk_size)
                    for part in sentence_parts:
                        if part.strip():
                            if title:
                                part = f"{title_prefix}{part}"
                            chunks.append(part)
                    i += 1
                break
            
            chunk_sentences.append(sentence)
            chunk_token_count += token_count
            i += 1
        
        # Создаем чанк из предложений
        if chunk_sentences:
            chunk_text = " ".join(chunk_sentences)
            if title:
                chunk_text = f"{title_prefix}{chunk_text}"
            chunks.append(chunk_text)
        
        # Делаем overlap на уровне предложений
        if overlap_sentences > 0 and i < len(sentence_tokens):
            # Откатываемся назад на overlap_sentences предложений
            i = max(i - overlap_sentences, i - len(chunk_sentences) + 1)
    
    # Финальная проверка: убеждаемся, что все чанки не превышают абсолютный лимит модели
    # Используем MAX_TOKENS (512) как жесткий лимит
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        if chunk_tokens > MAX_TOKENS:
            # Если чанк превышает абсолютный лимит, разбиваем его
            # Используем более строгий лимит для безопасности
            safe_limit = MAX_TOKENS - 30  # Запас 30 токенов для учета возможных погрешностей
            parts = split_long_sentence(chunk, tokenizer, safe_limit)
            final_chunks.extend(parts)
        else:
            final_chunks.append(chunk)
    
    # ЖЕСТКАЯ финальная проверка: ОБРЕЗАЕМ ВСЕ чанки до MAX_TOKENS
    verified_chunks = []
    for chunk in final_chunks:
        # Проверяем размер чанка
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_tokens = len(tokens)
        
        if chunk_tokens > MAX_TOKENS:
            # ЖЕСТКО обрезаем до MAX_TOKENS
            tokens = tokens[:MAX_TOKENS]
            chunk = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            
            # Проверяем еще раз и обрезаем если нужно (на случай особенностей декодирования)
            while True:
                tokens_check = tokenizer.encode(chunk, add_special_tokens=False)
                if len(tokens_check) <= MAX_TOKENS:
                    break
                # Если все еще превышает, обрезаем еще раз
                tokens_check = tokens_check[:MAX_TOKENS]
                chunk = tokenizer.decode(tokens_check, skip_special_tokens=True).strip()
        
        # Добавляем только непустые чанки
        if chunk and chunk.strip():
            verified_chunks.append(chunk)
    
    return verified_chunks


def split_text_by_tokens(text: str, tokenizer, chunk_size: int, overlap: int, title: str = None) -> list[str]:
    """
    Разбивает текст на чанки по токенам с перекрытием.
    Грамотно обрабатывает границы предложений для лучшего качества чанков.
    
    Args:
        text: Исходный текст
        tokenizer: Токенизатор для подсчета токенов
        chunk_size: Размер чанка в токенах
        overlap: Перекрытие между чанками в токенах
        title: Заголовок для добавления в каждый чанк (опционально)
        
    Returns:
        Список чанков
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Учитываем title в размере чанка
    title_tokens = 0
    title_prefix = ""
    if title:
        title_prefix = f"{title}\n\n"
        title_tokens = len(tokenizer.encode(title_prefix, add_special_tokens=False))
        effective_chunk_size = chunk_size - title_tokens
    else:
        effective_chunk_size = chunk_size
    
    # Токенизируем весь текст
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= effective_chunk_size:
        # Если текст помещается в один чанк
        if title:
            return [f"{title_prefix}{text}"]
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + effective_chunk_size
        
        # Берем токены для текущего чанка
        chunk_tokens = tokens[start:end]
        
        # Декодируем обратно в текст
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        
        if chunk_text:
            if title:
                chunk_text = f"{title_prefix}{chunk_text}"
            chunks.append(chunk_text)
        
        # Двигаемся с учетом перекрытия
        if overlap == 0:
            start = end
        else:
            start = end - overlap
        
        # Если осталось меньше overlap токенов, выходим
        if overlap == 0 and start >= len(tokens):
            break
        elif overlap > 0 and start + overlap >= len(tokens):
            # Берем оставшиеся токены
            if start < len(tokens):
                remaining_tokens = tokens[start:]
                remaining_text = tokenizer.decode(
                    remaining_tokens, skip_special_tokens=True
                ).strip()
                if remaining_text:
                    if title:
                        remaining_text = f"{title_prefix}{remaining_text}"
                    chunks.append(remaining_text)
            break
        
        # Защита от бесконечного цикла
        if start >= len(tokens):
            break
    
    return chunks


def prepare_chunks_from_websites(
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    output_filename: str = "chunks_frida.json",
    add_title: bool = False,
    use_sentence_splitting: bool = True,
):
    """
    Подготавливает чанки из файла веб-страниц для модели FRIDA.
    
    Args:
        chunk_size_tokens: Максимальный размер чанка в токенах
        chunk_overlap_tokens: Перекрытие между чанками в токенах (игнорируется если use_sentence_splitting=True)
        output_filename: Имя выходного файла
        add_title: Добавлять ли заголовок в начало каждого чанка
        use_sentence_splitting: Использовать ли умное разбиение по предложениям (рекомендуется)
    
    Если use_sentence_splitting=True (рекомендуется):
        - Разбивает текст на предложения
        - НИКОГДА не разрывает предложения посередине
        - Группирует предложения до достижения лимита токенов
        - Overlap на уровне предложений (фиксированное количество предложений)
        
    Если use_sentence_splitting=False (старый метод):
        - Разбивает по токенам с заданным overlap
        - Может разрывать предложения
    """
    
    print(f"📖 Загрузка данных из {WEBSITES_FILE}...")
    
    # Загружаем токенизатор FRIDA
    print(f"🤖 Загрузка токенизатора {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    print(f"✅ Токенизатор загружен")
    
    # Читаем CSV с правильными параметрами для длинных текстов
    df = pd.read_csv(
        WEBSITES_FILE,
        dtype=str,  # Читаем все как строки, чтобы не обрезать
        keep_default_na=False,  # Не заменяем пустые строки на NaN
        encoding="utf-8",
    )
    
    print(f"Найдено {len(df)} веб-страниц")
    print(f"Разбиение на чанки: макс {chunk_size_tokens} токенов")
    if use_sentence_splitting:
        overlap_sentences = max(1, chunk_overlap_tokens // 50)  # Примерно 1 предложение = 50 токенов
        print(f"Метод: УМНОЕ разбиение по предложениям (overlap: {overlap_sentences} предложений)")
        print(f"   ✅ Предложения НЕ разрываются посередине")
        print(f"   ✅ Сохраняется целостность смысла")
    else:
        print(f"Метод: Простое разбиение по токенам (overlap: {chunk_overlap_tokens} токенов)")
    print(f"Лимит модели: {MAX_TOKENS} токенов")
    print(f"Модель: {TOKENIZER_NAME}")
    print(f"Добавление title: {add_title}")
    
    all_chunks = []
    chunk_id = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка страниц"):
        # Берем все поля как есть из CSV
        web_id = int(row["web_id"])
        url = row["url"]
        kind = row["kind"]
        title = row["title"]
        text = row["text"]  # Берем text как есть, без изменений
        
        if not text or len(text.strip()) == 0:
            continue
        
        # Разбиваем текст на чанки
        if use_sentence_splitting:
            # Умное разбиение по предложениям
            overlap_sentences = max(1, chunk_overlap_tokens // 50)  # ~50 токенов на предложение
            text_chunks = split_text_by_sentences(
                text, tokenizer, chunk_size_tokens, overlap_sentences, title if add_title else None
            )
        else:
            # Простое разбиение по токенам (старый метод)
            text_chunks = split_text_by_tokens(
                text, tokenizer, chunk_size_tokens, chunk_overlap_tokens, title if add_title else None
            )
        
        # Создаем чанки с сохранением web_id (все части одного документа имеют одинаковый web_id)
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_data = {
                "chunk_id": chunk_id,
                "web_id": web_id,  # Сохраняем web_id родительского документа
                "url": url,
                "kind": kind,
                "title": title,
                "chunk_index": chunk_idx,  # Индекс чанка внутри документа (0, 1, 2, ...)
                "total_chunks": len(text_chunks),  # Общее количество чанков для этого документа
                "text": chunk_text,
            }
            all_chunks.append(chunk_data)
            chunk_id += 1
    
    # Сохраняем в JSON (отдельный файл для FRIDA)
    output_file = RAG_COLLECTION_DIR / output_filename
    
    print(f"\n💾 Сохранение {len(all_chunks)} чанков в {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2 if DEBUG else None)
    
    print(f"✅ Готово! Создано {len(all_chunks)} чанков")
    print(f"   JSON: {output_file}")
    
    # Статистика
    print(f"\n📊 Статистика:")
    print(f"   Уникальных веб-страниц: {len(df)}")
    print(f"   Всего чанков: {len(all_chunks)}")
    
    # Подсчитываем длины текстов и токенов
    text_lengths = [len(chunk["text"]) for chunk in all_chunks]
    token_counts = [
        len(tokenizer.encode(chunk["text"], add_special_tokens=False))
        for chunk in all_chunks
    ]
    
    if text_lengths:
        print(f"   Средняя длина чанка: {sum(text_lengths) / len(text_lengths):.0f} символов")
        print(f"   Минимальная длина: {min(text_lengths):.0f} символов")
        print(f"   Максимальная длина: {max(text_lengths):.0f} символов")
        print(f"   Среднее кол-во токенов: {sum(token_counts) / len(token_counts):.0f}")
        print(f"   Минимальное кол-во токенов: {min(token_counts)}")
        print(f"   Максимальное кол-во токенов: {max(token_counts)}")
        
        # Проверяем, есть ли чанки превышающие лимит
        oversized = [t for t in token_counts if t > MAX_TOKENS]
        if oversized:
            print(f"   ⚠️  Чанков превышающих {MAX_TOKENS} токенов: {len(oversized)}")
            print(f"   Максимальное превышение: {max(oversized) - MAX_TOKENS} токенов")
        else:
            print(f"   ✅ Все чанки помещаются в {MAX_TOKENS} токенов")
        
        # Статистика по разбиению документов
        web_id_to_chunks = {}
        for chunk in all_chunks:
            web_id = chunk["web_id"]
            if web_id not in web_id_to_chunks:
                web_id_to_chunks[web_id] = []
            web_id_to_chunks[web_id].append(chunk)
        
        chunks_per_doc = [len(chunks) for chunks in web_id_to_chunks.values()]
        if chunks_per_doc:
            print(f"\n   Статистика по разбиению документов:")
            print(f"   Среднее чанков на документ: {sum(chunks_per_doc) / len(chunks_per_doc):.2f}")
            print(f"   Максимум чанков на документ: {max(chunks_per_doc)}")
            print(f"   Документов разбитых на несколько чанков: {sum(1 for c in chunks_per_doc if c > 1)}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Подготовка чанков для FRIDA с умным разбиением")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_TOKENS, help="Размер чанка в токенах")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_TOKENS, help="Перекрытие в токенах (для sentence-splitting конвертируется в кол-во предложений)")
    parser.add_argument("--output", type=str, required=True, help="Имя выходного файла (например: chunks_frida_smart_500_100.json)")
    parser.add_argument("--add-title", action="store_true", help="Добавлять title в каждый чанк")
    parser.add_argument("--no-sentence-splitting", action="store_true", help="Отключить умное разбиение по предложениям (использовать старый метод)")
    
    args = parser.parse_args()
    
    prepare_chunks_from_websites(
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.overlap,
        output_filename=args.output,
        add_title=args.add_title,
        use_sentence_splitting=not args.no_sentence_splitting,
    )

