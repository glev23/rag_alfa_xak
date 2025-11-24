# 🚀 Быстрый старт

Минимальная инструкция для запуска системы.

## Установка (5 минут)

### Windows
```powershell
# 1. Создание окружения
python -m venv .venv
.venv\Scripts\activate

# 2. Установка зависимостей
pip install -r requirements.txt
```

### Linux/macOS
```bash
# 1. Создание окружения
python3 -m venv .venv
source .venv/bin/activate

# 2. Установка зависимостей
pip install -r requirements.txt
```

## Настройка

Отредактируйте `.env` и укажите ваш Hugging Face токен:
```bash
HF_TOKEN=YOUR_HF_TOKEN
```

Получить токен: https://huggingface.co/settings/tokens

## Запуск (3 команды)

### 1. Запустите Qdrant
```bash
docker compose up -d qdrant
```

### 2. Подготовьте чанки
```bash
python -m src.prepare_chunks_frida --chunk-size 500 --overlap 100 --output chunks_frida_500_100.json
```

### 3. Создайте эмбеддинги
```bash
python -m src.ingest_frida --input chunks_frida_500_100.json --collection bank_site_frida_500_100
```

### 4. Выполните поиск
```bash
python -m src.rag_retrieval_frida --collection bank_site_frida_500_100
```

## Результат

Файл с результатами будет сохранен в:
```
submissions/submission_frida_512_TIMESTAMP.csv
```

## Требования

- ✅ Python 3.10+
- ✅ Docker (для Qdrant)
- ✅ 8+ GB RAM
- ⚡ GPU рекомендуется (но не обязательно)

## Проблемы?

См. раздел **Troubleshooting** в основном [README.md](README.md)

