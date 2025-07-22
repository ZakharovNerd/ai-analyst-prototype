import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from dotenv import load_dotenv
from src.analytics_agent import AnalyticsAgent

load_dotenv()

def test_analytics_queries():
    agent = AnalyticsAgent(os.getenv("OPENAI_API_KEY"))
    
    test_queries = [
        "Посчитай количество активных пользователей по регионам за июнь 2024",
        "Какая конверсия пользователей из регистрации в покупку за июнь?",
        "Выведи средний чек заказа по каждому региону за июнь",
        "Сколько пользователей не делали заказы после регистрации в июне?",
        "Покажи топ-3 региона по количеству регистраций за июнь",
        "Какая доля отмененных заказов за июнь 2024?",
        "Посчитай LTV (lifetime value) на пользователя за июнь",
        "Какой процент пользователей сделал повторные покупки в июне?",
        "Выведи динамику регистраций по дням за июнь",
        "Сколько пользователей за июнь заходили на сайт, но не совершили покупок?"
    ]
    
    print("=== Тестирование аналитических запросов ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Запрос {i}: {query}")
        print("-" * 50)
        
        try:
            answer = agent.process_query(query)
            print(f"Ответ: {answer}")
        except Exception as e:
            print(f"Ошибка: {e}")
        
        print("=" * 70)
        print()

if __name__ == "__main__":
    test_analytics_queries()