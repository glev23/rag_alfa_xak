# 🚀 RAG Система для поиска по банковским документам

Финальное решение RAG системы на основе модели **FRIDA** (ai-forever/FRIDA) для семантического поиска по банковским документам.

## 📋 Содержание

- [Описание решения](#описание-решения)
- [Требования](#требования)
- [Установка](#установка)
  - [Windows](#windows)
  - [Linux/macOS](#linuxmacos)
- [Запуск](#запуск)
- [Структура проекта](#структура-проекта)
- [Параметры конфигурации](#параметры-конфигурации)

## 🎯 Описание решения

Данное решение использует:
- **Модель эмбеддингов**: FRIDA (ai-forever/FRIDA)
- **Размерность векторов**: 1536
- **Векторная БД**: Qdrant
- **Размер чанков**: 500 токенов с overlap 100 токенов
- **Умное разбиение**: по предложениям (без разрыва предложений посередине)
- **Top-K результатов**: 5 документов

### Основные особенности:

✅ **Умное разбиение текста** - предложения не разрываются посередине  
✅ **Оптимальный размер чанков** - 500 токенов (макс лимит модели 512)  
✅ **Грамотный overlap** - 100 токенов для сохранения контекста  
✅ **Агрегация по web_id** - для каждого документа берется лучший score  
✅ **Префиксы FRIDA** - `search_document:` для документов, `search_query:` для запросов  

## 📦 Требования

- **Python**: 3.10 или выше
- **Docker**: для запуска Qdrant
- **GPU**: рекомендуется (CUDA) для ускорения работы
- **RAM**: минимум 8 GB, рекомендуется 16 GB
- **Disk space**: ~5 GB для моделей и данных

## 🛠️ Установка

### Шаг 1: Создание виртуального окружения

#### Windows

```powershell
# Создание виртуального окружения
python -m venv .venv

# Активация окружения
.venv\Scripts\activate
```

#### Linux/macOS

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация окружения
source .venv/bin/activate
```

### Шаг 2: Установка зависимостей

```bash
pip install -r requirements.txt
```

### Шаг 3: Настройка конфигурации

Отредактируйте файл `.env` и укажите свой Hugging Face токен:

```bash
# .env
HF_TOKEN= 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
```

> **Примечание**: Токен нужен для загрузки модели FRIDA с Hugging Face.  
> Получить токен можно на https://huggingface.co/settings/tokens

### Шаг 4: Подготовка данных

Разместите файлы данных в папке `data/`:
- `websites_cleaned.csv` - очищенные веб-страницы
- `questions_clean.csv` - вопросы для поиска

## 🚀 Запуск

### 1. Запуск Qdrant (векторная БД)

#### Windows

```powershell
# Запуск Qdrant в фоновом режиме
docker compose up -d qdrant
```

#### Linux/macOS

```bash
# Запуск Qdrant в фоновом режиме
docker compose up -d qdrant
```

**Проверка запуска:**
- Web UI: http://localhost:6883/dashboard
- API: http://localhost:6883

### 2. Подготовка чанков

Разбиваем тексты на чанки по 500 токенов с overlap 100:

```bash
python -m src.prepare_chunks_frida --chunk-size 500 --overlap 100 --output chunks_frida_500_100.json
```

**Параметры:**
- `--chunk-size` - размер чанка в токенах (рекомендуется 500)
- `--overlap` - перекрытие между чанками в токенах (рекомендуется 100)
- `--output` - имя выходного файла
- `--add-title` (опционально) - добавлять заголовок в каждый чанк
- `--no-sentence-splitting` (опционально) - отключить умное разбиение по предложениям

**Результат:**
- Файл `data/rag_collection/chunks_frida_500_100.json` с чанками
- Все чанки гарантированно ≤ 512 токенов (лимит FRIDA)

### 3. Создание эмбеддингов и загрузка в Qdrant

```bash
python -m src.ingest_frida --input chunks_frida_500_100.json --collection bank_site_frida_500_100
```

**Параметры:**
- `--input` - имя файла с чанками (из `data/rag_collection/`)
- `--collection` - название коллекции в Qdrant

**Процесс:**
- Загрузка модели FRIDA
- Создание эмбеддингов с префиксом `search_document:`
- Загрузка в Qdrant коллекцию
- Время: ~30-60 минут (зависит от GPU)

### 4. Поиск ответов

```bash
python -m src.rag_retrieval_frida --collection bank_site_frida_500_100 --batch-size 32 --top-k 5
```

**Параметры:**
- `--collection` - название коллекции в Qdrant
- `--batch-size` - размер батча для обработки (по умолчанию 32)
- `--top-k` - количество возвращаемых документов (по умолчанию 5)
- `--model-name` - имя для submission файла (по умолчанию "frida_512")

**Результат:**
- Файл `submissions/submission_frida_512_TIMESTAMP.csv`
- Формат: `q_id,web_list` где web_list это список из 5 web_id

## 📁 Структура проекта

```
rag_xakaton_final/
├── src/
│   ├── config.py                    # Конфигурация проекта
│   ├── prepare_chunks_frida.py      # Подготовка чанков
│   ├── ingest_frida.py              # Создание эмбеддингов и загрузка в Qdrant
│   └── rag_retrieval_frida.py       # Поиск и создание submission
├── data/
│   ├── rag_collection/              # JSON файлы с чанками
│   ├── websites_cleaned.csv         # Исходные документы
│   └── questions_clean.csv          # Вопросы
├── submissions/                     # Результаты поиска
├── qdrant/
│   └── storage/                     # Хранилище Qdrant
├── .env                             # Переменные окружения
├── docker-compose.yml               # Конфигурация Docker
├── requirements.txt                 # Зависимости проекта
└── README.md                        # Документация
```

## ⚙️ Параметры конфигурации

### Файл `.env`

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6883
QDRANT_GRPC_PORT=6884

# Hugging Face Token
HF_TOKEN=your_token_here

# Embedding Models Configuration
EMBEDDING_DIMENSION_FRIDA=1024

# RAG параметры
TOP_K_RESULTS=5

# Paths
DATA_DIR=data
RAG_COLLECTION_DIR=data/rag_collection
SUBMISSION_DIR=submissions

# Logging
DEBUG=True
```

### Оптимальные параметры (протестировано):

| Параметр | Значение | Описание |
|----------|----------|----------|
| Chunk Size | 500 | Размер чанка в токенах |
| Overlap | 100 | Перекрытие между чанками |
| Top-K | 5 | Количество результатов |
| Batch Size | 32 | Размер батча для обработки |
| Model | FRIDA | ai-forever/FRIDA |
| Splitting | Sentence-based | Умное разбиение по предложениям |

## 🔍 Полный пайплайн (краткая версия)

```bash
# 1. Запуск Qdrant
docker compose up -d qdrant

# 2. Активация окружения
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Подготовка чанков
python -m src.prepare_chunks_frida --chunk-size 500 --overlap 100 --output chunks_frida_500_100.json

# 4. Создание эмбеддингов
python -m src.ingest_frida --input chunks_frida_500_100.json --collection bank_site_frida_500_100

# 5. Поиск
python -m src.rag_retrieval_frida --collection bank_site_frida_500_100
```

## 🐛 Troubleshooting

### Ошибка: "CUDA out of memory"

**Решение:**
- Уменьшите batch_size при ingest: используйте обработку по одному документу (уже настроено)
- Используйте CPU: модель автоматически переключится на CPU если GPU недоступен

### Ошибка: "Collection not found"

**Решение:**
- Убедитесь, что вы запустили `ingest_frida.py` перед `rag_retrieval_frida.py`
- Проверьте название коллекции (должно совпадать в обеих командах)

### Ошибка: "File not found: websites_cleaned.csv"

**Решение:**
- Разместите файлы данных в папке `data/`
- Проверьте пути в `.env` файле

### Qdrant не запускается

**Решение Windows:**
```powershell
# Проверка Docker
docker --version

# Перезапуск Qdrant
docker compose down
docker compose up -d qdrant
```

**Решение Linux/macOS:**
```bash
# Проверка Docker
docker --version

# Перезапуск Qdrant
docker compose down
docker compose up -d qdrant
```

## 📊 Ожидаемые результаты

- **Количество чанков**: ~10,000-15,000 (зависит от данных)
- **Размер коллекции**: ~2-3 GB
- **Время создания эмбеддингов**: 30-60 минут (GPU) / 2-4 часа (CPU)
- **Время поиска**: 2-5 минут для всех вопросов
- **Точность**: зависит от качества данных

## 👥 Авторы

- glev23 https://t.me/korcy_lives

