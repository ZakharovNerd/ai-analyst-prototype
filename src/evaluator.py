import os
import json
from langsmith import Client
from langchain.smith import run_on_dataset
from langchain_openai import ChatOpenAI
from .analytics_agent import AnalyticsAgent
import time
import logging

logger = logging.getLogger(__name__)

class AnswerEvaluator:
    def __init__(self, openai_api_key: str, langsmith_api_key: str | None = None):
        self.client = Client(api_key=langsmith_api_key)
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        
    def extract_user_request(self, input_data):
        if isinstance(input_data, dict):
            if "messages" in input_data:
                messages = input_data["messages"]
            elif "inputs" in input_data and isinstance(input_data["inputs"], list):
                messages = input_data["inputs"]
            elif "input" in input_data and isinstance(input_data["input"], list):
                messages = input_data["input"]
            else:
                return input_data.get("input") or input_data.get("question") or next(iter(input_data.values()), "")
            
            user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
            if user_msg is not None:
                return user_msg
            system_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "system"), "")
            return system_msg
        elif isinstance(input_data, list):
            user_msg = next((m["content"] for m in reversed(input_data) if m.get("role") == "user"), None)
            if user_msg is not None:
                return user_msg
            system_msg = next((m["content"] for m in reversed(input_data) if m.get("role") == "system"), "")
            return system_msg
        else:
            return str(input_data)
    
    def process_input(self, input_data):
        try:
            analyst = AnalyticsAgent(self.openai_api_key)
            user_request = self.extract_user_request(input_data)
            
            from .analytics_agent import AnalyticsState
            initial_state = AnalyticsState(user_query=user_request)
            final_state = analyst.graph.invoke(initial_state)
            
            if hasattr(final_state, 'final_answer'):
                result = final_state.final_answer or "Не удалось обработать запрос"
                code_reasoning = getattr(final_state, 'code_reasoning', '')
                answer_reasoning = getattr(final_state, 'answer_reasoning', '')
                pandas_code = getattr(final_state, 'pandas_code', '')
                execution_result = getattr(final_state, 'execution_result', '')
            else:
                result = final_state.get('final_answer') or "Не удалось обработать запрос"
                code_reasoning = final_state.get('code_reasoning', '')
                answer_reasoning = final_state.get('answer_reasoning', '')
                pandas_code = final_state.get('pandas_code', '')
                execution_result = final_state.get('execution_result', '')
            
            time.sleep(0.02)
            
            data_schema = analyst.data_processor.get_data_schema()
            
            return {
                "answer": result,
                "code_reasoning": code_reasoning,
                "answer_reasoning": answer_reasoning,
                "pandas_code": pandas_code,
                "execution_result": execution_result,
                "data_schema": data_schema,
                "success": True
            }
        except Exception as e:
            logger.exception("Error processing input for evaluation")
            return {
                "answer": "",
                "error": str(e),
                "success": False
            }
    
    def run_evaluation(self, dataset_name: str, project_name: str = "analytics-bot-eval"):
        evaluation_config = {
            "criteria": ["helpfulness", "accuracy"],
            "llm": self.llm
        }
        
        run_on_dataset(
            client=self.client,
            dataset_name=dataset_name,
            llm_or_chain_factory=self.process_input,
            evaluation_config=evaluation_config,
            project_name=project_name
        )