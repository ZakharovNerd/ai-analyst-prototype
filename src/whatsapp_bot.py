from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from .analytics_agent import AnalyticsAgent
import logging

logger = logging.getLogger(__name__)

class WhatsAppBot:
    def __init__(self, account_sid: str, auth_token: str, phone_number: str, openai_api_key: str):
        self.client = Client(account_sid, auth_token)
        self.phone_number = phone_number
        self.analytics_agent = AnalyticsAgent(openai_api_key)
    
    def handle_message(self, from_number: str, message_body: str) -> str:
        try:
            if not message_body.strip():
                return "Пожалуйста, задайте вопрос для аналитики данных."
            
            answer = self.analytics_agent.process_query(message_body)
            return answer
            
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