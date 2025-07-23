# AI Analytics Assistant

WhatsApp-интегрированный AI ассистент для бизнес-аналитики с использованием LangGraph и автоматической оценкой через LangSmith.

## Возможности

- 📊 Text-to-pandas конвертация запросов с retry логикой
- 📱 WhatsApp интеграция через Twilio
- 🔄 LangGraph workflow для обработки запросов
- ⚡ FastAPI REST API
- 📈 LangSmith evaluation и мониторинг

## Установка

```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # или ./venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Заполните .env своими ключами
```

## Тестирование

### 1. Локальное тестирование всех запросов
```bash
./venv/bin/python tests/test_queries.py
```

### 2. API тестирование
```bash
# Запуск сервера
./venv/bin/python app.py

# В другом терминале
curl -X POST "http://localhost:8000/test/query" \
     -H "Content-Type: application/json" \
     -d '{"message": "Посчитай активных пользователей за июнь 2024"}'
```

### 3. LangSmith оценка
```bash
# Создание датасета
./venv/bin/python tests/create_dataset.py

# Запуск оценки
./venv/bin/python tests/run_evaluation.py

# Проверка LangSmith подключения
./venv/bin/python tests/test_langsmith.py
```

## Архитектура

```
src/
├── analytics_agent.py    # LangGraph агент с text-to-pandas
├── data_processor.py     # Выполнение pandas кода
├── whatsapp_bot.py      # WhatsApp интеграция
└── evaluator.py         # LangSmith оценка

tests/
├── test_queries.py      # Тесты всех запросов
├── create_dataset.py    # Создание LangSmith датасета
├── run_evaluation.py    # Запуск оценки
└── test_langsmith.py    # Тест подключения LangSmith

data/
├── users.csv           # Данные пользователей (150 строк)
└── orders.csv          # Данные заказов (200 строк)
```
