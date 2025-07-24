import json
from typing import Dict, Any
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from .data_processor import DataProcessor
import logging

logger = logging.getLogger(__name__)

class QueryResponse(BaseModel):
    requires_code: bool = Field(description="Требует ли запрос выполнения pandas кода")
    reasoning: str = Field(description="Логика и рассуждения о том, как решить задачу")
    pandas_code: str | None = Field(description="Pandas код для выполнения (если requires_code=True)", default=None)
    direct_answer: str | None = Field(description="Прямой ответ без кода (если requires_code=False)", default=None)

class AnswerResponse(BaseModel):
    reasoning: str = Field(description="Анализ результатов и логика формирования ответа")
    final_answer: str = Field(description="Финальный ответ пользователю")

@dataclass
class AnalyticsState:
    user_query: str
    requires_data_analysis: bool | None = None
    pandas_code: str | None = None
    code_reasoning: str | None = None
    execution_result: Any = None
    execution_error: str | None = None
    final_answer: str | None = None
    answer_reasoning: str | None = None
    retry_count: int = 0
    max_retries: int = 3

class AnalyticsAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key
        )
        self.data_processor = DataProcessor()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AnalyticsState)
        
        workflow.add_node("query_processor", self._process_query)
        workflow.add_node("code_executor", self._execute_code)
        workflow.add_node("answer_formatter", self._format_answer)
        
        workflow.set_entry_point("query_processor")
        
        workflow.add_conditional_edges(
            "query_processor",
            self._route_after_query_processing,
            {
                "execute": "code_executor",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "code_executor",
            self._should_retry,
            {
                "retry": "query_processor",
                "format": "answer_formatter"
            }
        )
        
        workflow.add_edge("answer_formatter", END)
        
        return workflow.compile()
    
    def _process_query(self, state: AnalyticsState) -> AnalyticsState:
        data_schema = self.data_processor.get_data_schema()
        
        system_prompt = f"""Ты AI ассистент по аналитике данных. Определи, требует ли запрос анализа данных или это обычный вопрос.

Доступные данные для анализа:
{data_schema}

Если запрос требует анализа данных (requires_code=true):
- Вопросы о количестве, статистике, метриках
- Расчеты конверсии, LTV, средних значений  
- Группировки по регионам, датам
- Анализ пользователей, заказов

Правила для pandas кода:
1. Всегда присваивай финальный результат переменной 'result'
2. Используй только pandas операции
3. Для дат используй формат 2024-06-01 (год-месяц-день)
4. КРИТИЧНО: При работе с заказами учитывай статус - используй только 'completed' для расчетов доходов

Если запрос НЕ требует анализа данных (requires_code=false):
- Приветствия, благодарности
- Вопросы о возможностях бота
- Общие вопросы без привязки к данным

Сначала объясни логику, затем предоставь либо код, либо прямой ответ."""
        
        error_context = ""
        if state.execution_error and state.retry_count > 0:
            error_context = f"\nПредыдущая ошибка: {state.execution_error}\nИсправь код."
        
        messages = [
            SystemMessage(content=system_prompt + error_context),
            HumanMessage(content=state.user_query)
        ]
        
        structured_llm = self.llm.with_structured_output(QueryResponse)
        response = structured_llm.invoke(messages)
        
        state.requires_data_analysis = response.requires_code
        state.code_reasoning = response.reasoning
        
        if response.requires_code:
            state.pandas_code = response.pandas_code
        else:
            state.final_answer = response.direct_answer
        
        return state
    
    def _route_after_query_processing(self, state: AnalyticsState) -> str:
        return "execute" if state.requires_data_analysis else "end"
    
    def _execute_code(self, state: AnalyticsState) -> AnalyticsState:
        if not state.pandas_code:
            logger.warning("No pandas code generated for execution")
            state.execution_error = "No pandas code generated"
            return state
        
        logger.info(f"Executing pandas code (attempt {state.retry_count + 1}): {state.pandas_code[:100]}...")
        result, error = self.data_processor.execute_pandas_query(state.pandas_code)
        
        if error:
            logger.warning(f"Code execution failed (attempt {state.retry_count + 1}): {error}")
            state.execution_error = error
            state.retry_count += 1
        else:
            logger.info(f"Code execution successful. Result type: {type(result)}")
            # Convert pandas objects to a more manageable format
            if hasattr(result, 'to_dict'):
                state.execution_result = result.to_dict('records')
            elif hasattr(result, 'to_json'):
                state.execution_result = result.to_json(orient='split')
            else:
                state.execution_result = result
            state.execution_error = None
        
        return state
    
    def _should_retry(self, state: AnalyticsState) -> str:
        if state.execution_error and state.retry_count < state.max_retries:
            logger.info(f"Retrying code generation (attempt {state.retry_count + 1}/{state.max_retries})")
            return "retry"
        
        if state.execution_error:
            logger.error(f"Max retries exceeded. Final error: {state.execution_error}")
        else:
            logger.info("Code execution successful, proceeding to answer formatting")
        
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
6. КРИТИЧНО: Используй точку (.) как десятичный разделитель (например, 25.4%).
7. КРИТИЧНО: Не используй специальные символы (например, →, ₽). Заменяй их текстом (например, '->', 'руб.').

Примеры хороших ответов:
- "Активные пользователи по регионам за июнь 2024: Москва - 15, СПб - 12, Казань - 8"

Сначала проанализируй данные и логику ответа, затем дай финальный ответ."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Пользовательский запрос: {state.user_query}\n\nРезультат pandas: {state.execution_result}")
        ]
        
        structured_llm = self.llm.with_structured_output(AnswerResponse)
        response = structured_llm.invoke(messages)
        
        state.final_answer = response.final_answer
        state.answer_reasoning = response.reasoning
        
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