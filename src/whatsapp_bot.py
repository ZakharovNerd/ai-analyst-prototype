from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from .analytics_agent import AnalyticsAgent, AnalyticsState
from .answer_evaluator import AnswerEvaluator
import logging

logger = logging.getLogger(__name__)

class WhatsAppBot:
    def __init__(self, account_sid: str, auth_token: str, phone_number: str, openai_api_key: str):
        self.client = Client(account_sid, auth_token)
        self.phone_number = phone_number
        self.analytics_agent = AnalyticsAgent(openai_api_key)
        self.answer_evaluator = AnswerEvaluator(openai_api_key)
    
    def handle_message(self, from_number: str, message_body: str) -> str:
        try:
            if not message_body.strip():
                return "Пожалуйста, задайте вопрос для аналитики данных."
            
            # Получаем детальный результат через граф
            initial_state = AnalyticsState(user_query=message_body)
            final_state = self.analytics_agent.graph.invoke(initial_state)
            
            if hasattr(final_state, 'final_answer'):
                answer = final_state.final_answer or "Не удалось обработать запрос"
                pandas_code = getattr(final_state, 'pandas_code', '')
                execution_result = getattr(final_state, 'execution_result', '')
            else:
                answer = final_state.get('final_answer') or "Не удалось обработать запрос"
                pandas_code = final_state.get('pandas_code', '')
                execution_result = final_state.get('execution_result', '')
            
            # Получаем оценку с полным контекстом
            code_reasoning = getattr(final_state, 'code_reasoning', '') if hasattr(final_state, 'code_reasoning') else final_state.get('code_reasoning', '')
            answer_reasoning = getattr(final_state, 'answer_reasoning', '') if hasattr(final_state, 'answer_reasoning') else final_state.get('answer_reasoning', '')
            
            evaluation = self.answer_evaluator.evaluate_answer(
                message_body, answer, pandas_code, str(execution_result), code_reasoning, answer_reasoning
            )
            
            # Формируем ответ с оценкой
            response = f"Запрос:\n«{message_body}»\n\nОтвет:\n{answer}\n\nEval (автоматическая оценка соответствия ответа запросу):\n{evaluation['evaluation_text']}"
            
            return response
            
        except Exception:
            logger.exception("Error handling WhatsApp message")
            return "Произошла ошибка при обработке сообщения."
    
    def send_message(self, to_number: str, message: str):
        try:
            self.client.messages.create(
                from_=f'whatsapp:{self.phone_number}',
                body=message,
                to=f'whatsapp:{to_number}'
            )
        except Exception:
            logger.exception("Failed to send WhatsApp message")
    
    def create_webhook_response(self, message: str) -> str:
        response = MessagingResponse()
        response.message(message)
        return str(response)