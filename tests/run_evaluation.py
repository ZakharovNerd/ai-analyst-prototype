import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from src.evaluator import AnswerEvaluator

load_dotenv()

def run_evaluation():
    evaluator = AnswerEvaluator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        langsmith_api_key=os.getenv("LANGCHAIN_API_KEY")
    )
    
    dataset_name = "vividmoney-analytics"
    project_name = "vividmoney-evaluation-run"
    
    print(f"Запускаем оценку на датасете: {dataset_name}")
    print(f"Проект: {project_name}")
    
    try:
        evaluator.run_evaluation(dataset_name, project_name)
        print("Оценка завершена!")
        print(f"Результаты доступны в LangSmith проекте: {project_name}")
    except Exception as e:
        print(f"Ошибка при запуске оценки: {e}")

if __name__ == "__main__":
    run_evaluation()