import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

def create_test_dataset():
    client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
    
    dataset_name = "vividmoney-analytics"
    
    test_examples = [
        {"input": "Посчитай количество активных пользователей по регионам за июнь 2024"},
        {"input": "Какая конверсия пользователей из регистрации в покупку за июнь?"},
        {"input": "Выведи средний чек заказа по каждому региону за июнь"},
        {"input": "Сколько пользователей не делали заказы после регистрации в июне?"},
        {"input": "Покажи топ-3 региона по количеству регистраций за июнь"},
        {"input": "Какая доля отмененных заказов за июнь 2024?"},
        {"input": "Посчитай LTV (lifetime value) на пользователя за июнь"},
        {"input": "Какой процент пользователей сделал повторные покупки в июне?"},
        {"input": "Выведи динамику регистраций по дням за июнь"},
        {"input": "Сколько пользователей за июнь заходили на сайт, но не совершили покупок?"}
    ]
    
    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name, 
            description="VividMoney Analytics Bot Test Dataset - All 10 queries"
        )
        print(f"Создан датасет: {dataset_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Датасет {dataset_name} уже существует")
            dataset = client.read_dataset(dataset_name=dataset_name)
        else:
            raise e
    
    for i, example in enumerate(test_examples):
        try:
            client.create_example(
                inputs=example,
                dataset_id=dataset.id
            )
            print(f"Добавлен пример {i+1}: {example['input'][:50]}...")
        except Exception as e:
            print(f"Пример {i+1} уже существует или ошибка: {e}")
    
    return dataset_name

if __name__ == "__main__":
    dataset_name = create_test_dataset()
    print(f"\nДатасет готов: {dataset_name}")