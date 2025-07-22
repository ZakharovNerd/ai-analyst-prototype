import json
from typing import Dict, Any
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .data_processor import DataProcessor
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsState:
    user_query: str
    pandas_code: str | None = None
    execution_result: Any = None
    execution_error: str | None = None
    final_answer: str | None = None
    retry_count: int = 0
    max_retries: int = 3

class AnalyticsAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        self.data_processor = DataProcessor()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AnalyticsState)
        
        workflow.add_node("code_generator", self._generate_pandas_code)
        workflow.add_node("code_executor", self._execute_code)
        workflow.add_node("answer_formatter", self._format_answer)
        
        workflow.set_entry_point("code_generator")
        workflow.add_edge("code_generator", "code_executor")
        
        workflow.add_conditional_edges(
            "code_executor",
            self._should_retry,
            {
                "retry": "code_generator",
                "format": "answer_formatter"
            }
        )
        
        workflow.add_edge("answer_formatter", END)
        
        return workflow.compile()
    
    def _generate_pandas_code(self, state: AnalyticsState) -> AnalyticsState:
        data_schema = self.data_processor.get_data_schema()
        
        system_prompt = f"""Ты эксперт по pandas. Преобразуй пользовательский запрос в код pandas.

Доступные данные:
{data_schema}

ВАЖНО:
1. Всегда присваивай финальный результат переменной 'result'
2. Используй только pandas операции
3. Для дат используй формат 2024-06-01 (год-месяц-день)
4. Для фильтрации по июню 2024: df['date_column'].dt.month == 6 & df['date_column'].dt.year == 2024
5. Не используй импорты - pandas уже доступен как 'pd'
6. Возвращай только код, без объяснений

Примеры:
- "активные пользователи по регионам за июнь" → june_users = users_df[(users_df['registration_date'].dt.month == 6) & (users_df['registration_date'].dt.year == 2024) & (users_df['is_active'] == True)]
result = june_users.groupby('region').size()"""
        
        error_context = ""
        if state.execution_error and state.retry_count > 0:
            error_context = f"\nПредыдущая ошибка: {state.execution_error}\nИсправь код."
        
        messages = [
            SystemMessage(content=system_prompt + error_context),
            HumanMessage(content=state.user_query)
        ]
        
        response = self.llm.invoke(messages)
        
        code = response.content.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        state.pandas_code = code
        return state
    
    def _execute_code(self, state: AnalyticsState) -> AnalyticsState:
        if not state.pandas_code:
            state.execution_error = "No pandas code generated"
            return state
        
        result, error = self.data_processor.execute_pandas_query(state.pandas_code)
        
        if error:
            state.execution_error = error
            state.retry_count += 1
        else:
            state.execution_result = result
            state.execution_error = None
        
        return state
    
    def _should_retry(self, state: AnalyticsState) -> str:
        if state.execution_error and state.retry_count < state.max_retries:
            return "retry"
        return "format"
    
    def _format_answer(self, state: AnalyticsState) -> AnalyticsState:
        if state.execution_error:
            state.final_answer = f"Ошибка при выполнении запроса: {state.execution_error}"
            return state
        
        system_prompt = """Ты аналитик данных. Преобразуй результат pandas запроса в краткий понятный ответ для пользователя.

Правила:
1. Отвечай кратко и по существу
2. Используй конкретные цифры
3. Добавляй контекст если нужно
4. Не объясняй как получен результат
5. Формат: одно-два предложения максимум

Примеры хороших ответов:
- "Активные пользователи по регионам за июнь 2024: Москва - 15, СПб - 12, Казань - 8"
- "Конверсия регистрации → покупка за июнь 2024: 25,4% (254 из 1000 пользователей)"
- "Средний чек по регионам за июнь: Москва - 8500₽, СПб - 7200₽, Казань - 6800₽"
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Пользовательский запрос: {state.user_query}\n\nРезультат pandas: {state.execution_result}")
        ]
        
        response = self.llm.invoke(messages)
        state.final_answer = response.content.strip()
        
        return state
    
    def process_query(self, user_query: str) -> str:
        try:
            initial_state = AnalyticsState(user_query=user_query)
            final_state = self.graph.invoke(initial_state)
            
            if hasattr(final_state, 'final_answer'):
                return final_state.final_answer or "Не удалось обработать запрос"
            else:
                return final_state.get('final_answer') or "Не удалось обработать запрос"
            
        except Exception:
            logger.exception("Error processing query")
            return "Произошла ошибка при обработке запроса"