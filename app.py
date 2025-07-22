from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
import os
from dotenv import load_dotenv
from src.whatsapp_bot import WhatsAppBot
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VividMoney Analytics Bot")

bot = WhatsAppBot(
    account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
    auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
    phone_number=os.getenv("TWILIO_PHONE_NUMBER"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "VividMoney Analytics Bot"}

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, Body: str = Form(...), From: str = Form(...)):
    try:
        logger.info(f"Received message from {From}: {Body}")
        
        response_message = bot.handle_message(From, Body)
        twiml_response = bot.create_webhook_response(response_message)
        
        return Response(content=twiml_response, media_type="application/xml")
        
    except Exception:
        logger.exception("Error processing WhatsApp webhook")
        error_response = bot.create_webhook_response("Произошла ошибка при обработке сообщения.")
        return Response(content=error_response, media_type="application/xml")

@app.post("/test/query")
async def test_query(query: dict):
    try:
        user_query = query.get("message", "")
        if not user_query:
            return {"error": "Message field is required"}
        
        response = bot.handle_message("test_user", user_query)
        return {"response": response}
        
    except Exception:
        logger.exception("Error processing test query")
        return {"error": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)