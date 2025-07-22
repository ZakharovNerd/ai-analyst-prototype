# VividMoney AI Analytics Assistant

## What I'm Working On

Creating a WhatsApp-integrated AI analytics assistant using LangGraph that processes business analytics queries from CSV data.

## Project Structure

- `data/` - Contains users.csv (150 rows) and orders.csv (200 rows) with sample e-commerce data
- `analytics_agent.py` - LangGraph-based agent for processing analytics queries
- `data_processor.py` - CSV data analysis functions  
- `whatsapp_bot.py` - WhatsApp integration using Twilio
- `evaluator.py` - Automatic answer quality evaluation
- `app.py` - Main FastAPI application

## Key Components

1. **LangGraph Agent**: Text-to-pandas agent that converts natural language queries to pandas code with retry logic
2. **Data Processor**: Dynamic pandas query executor with error handling and code generation
3. **WhatsApp Integration**: Twilio webhook for message handling
4. **Auto-Evaluation**: LangSmith-based evaluation scoring answer quality

## Sample Queries Supported

- Active users by region for June 2024
- Registration to purchase conversion rate
- Average order value by region
- User retention metrics
- Order cancellation rates
- LTV calculations

## Tech Stack

- LangGraph for agent workflow
- pandas for data analysis
- FastAPI for webhook server
- Twilio for WhatsApp integration
- OpenAI API for LLM capabilities