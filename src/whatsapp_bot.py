from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from .analytics_agent import AnalyticsAgent, AnalyticsState
from .answer_evaluator import AnswerEvaluator
import logging
from twilio.base.exceptions import TwilioRestException

logger = logging.getLogger(__name__)

class WhatsAppBot:
    def __init__(self, account_sid: str, auth_token: str, phone_number: str, openai_api_key: str):
        self.client = Client(account_sid, auth_token)
        self.phone_number = phone_number
        self.analytics_agent = AnalyticsAgent(openai_api_key)
        self.answer_evaluator = AnswerEvaluator(openai_api_key)
    
    def handle_message(self, from_number: str, message_body: str) -> str:
        try:
            logger.info(f"Received message from {from_number}: '{message_body[:100]}...'")
            
            if not message_body.strip():
                return "Пожалуйста, задайте вопрос для аналитики данных."
            
            # Получаем детальный результат через граф
            initial_state = AnalyticsState(user_query=message_body)
            logger.info(f"Invoking analytics graph for query: '{message_body[:50]}...'")
            
            final_state = self.analytics_agent.graph.invoke(initial_state)
            
            logger.info(f"Graph execution completed. State keys: {list(final_state.keys())}")
            logger.info(f"requires_data_analysis: {final_state.get('requires_data_analysis')}")
            
            # Граф возвращает dict, а не dataclass
            answer = final_state.get('final_answer') or "Не удалось обработать запрос"
            pandas_code = final_state.get('pandas_code', '')
            execution_result = final_state.get('execution_result', '')
            
            if not answer or answer == "Не удалось обработать запрос":
                logger.warning(f"No valid answer generated. Final state: {final_state}")
            
            logger.info(f"Answer generated: '{answer[:100]}...' (length: {len(answer)})")
            logger.info(f"Pandas code present: {bool(pandas_code)}")
            logger.info(f"Execution result present: {bool(execution_result)}")
            
            # Получаем оценку с полным контекстом
            code_reasoning = final_state.get('code_reasoning', '')
            answer_reasoning = final_state.get('answer_reasoning', '')
            
            logger.info("Starting answer evaluation...")
            evaluation = self.answer_evaluator.evaluate_answer(
                message_body, answer, pandas_code, str(execution_result), code_reasoning, answer_reasoning
            )
            
            # Логируем и принтуем детальные результаты оценки
            eval_log = f"Query: {message_body[:50]}..."
            eval_log += f"\nCorrectness: {evaluation['correctness']}/5 - {evaluation['correctness_reasoning'][:100]}..."
            eval_log += f"\nConciseness: {evaluation['conciseness']}/5 - {evaluation['conciseness_reasoning'][:100]}..."
            eval_log += f"\nCode Quality: {evaluation['code_checker']}/5 - {evaluation['code_reasoning'][:100]}..."
            eval_log += f"\nOverall: {evaluation['overall_score']}/5"
            
            logger.info(f"Evaluation completed:\n{eval_log}")
            
            # Подготавливаем текст оценки для включения в основной ответ
            evaluation_text_for_display = ""
            if final_state.get('requires_data_analysis') and evaluation and evaluation.get('overall_score') is not None:
                evaluation_text_for_display = f"\n\nEval (автоматическая оценка соответствия ответа запросу):\n{evaluation['evaluation_text']}"
            
            final_response_content = f"{answer}{evaluation_text_for_display}"

            logger.info(f"Final response content ready (length: {len(final_response_content)})")
            
            return final_response_content
            
        except Exception as e:
            logger.exception("Error handling WhatsApp message")
            error_message = "Произошла ошибка при обработке сообщения."
            if isinstance(e, TwilioRestException) and e.status == 429:
                error_message = "Извините, превышен лимит на количество сообщений Twilio Sandbox. Пожалуйста, попробуйте позже."
            return error_message
    
    def send_message(self, to_number: str, message: str):
        try:
            self.client.messages.create(
                from_=f'whatsapp:{self.phone_number}',
                body=message,
                to=f'whatsapp:{to_number}'
            )

        except TwilioRestException as e:
            logger.exception(f"Failed to send WhatsApp message: {e}")
            raise e # Re-raise to be caught by handle_message
        except Exception as e:
            logger.exception(f"An unexpected error occurred while sending WhatsApp message: {e}")
            raise e
    
    def create_webhook_response(self, message: str) -> str:
        response = MessagingResponse()
        response.message(message)
        return str(response)
