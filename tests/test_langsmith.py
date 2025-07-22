import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from src.evaluator import AnswerEvaluator

load_dotenv()

def test_langsmith():
    evaluator = AnswerEvaluator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        langsmith_api_key=os.getenv("LANGCHAIN_API_KEY")
    )
    
    print("Тестируем LangSmith evaluation...")
    
    # Простой тест
    test_input = {"question": "Посчитай активных пользователей по регионам за июнь 2024"}
    result = evaluator.process_input(test_input)
    
    print(f"Результат: {result}")

if __name__ == "__main__":
    test_langsmith()