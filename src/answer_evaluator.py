from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class EvaluationResult:
    def __init__(self, score: int, reasoning: str):
        self.score = score
        self.reasoning = reasoning

class AnswerEvaluator:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key
        )
    
    def evaluate_answer(self, user_query: str, answer: str, pandas_code: str = "", execution_result: str = "", code_reasoning: str = "", answer_reasoning: str = "") -> Dict[str, Any]:
        """Оценивает ответ по 3 критериям: correctness, conciseness, code_checker"""
        
        correctness_result = self._evaluate_correctness(user_query, answer, execution_result, answer_reasoning)
        conciseness_result = self._evaluate_conciseness(user_query, answer, answer_reasoning)
        
        # Оценка кода только если есть pandas_code
        if pandas_code and pandas_code.strip():
            code_checker_result = self._evaluate_code_quality(pandas_code, user_query, code_reasoning)
            scores = [correctness_result.score, conciseness_result.score, code_checker_result.score]
        else:
            code_checker_result = None
            scores = [correctness_result.score, conciseness_result.score]
        
        # Общая оценка (среднее арифметическое только для не-None значений)
        valid_scores = [s for s in scores if s is not None]
        overall_score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else None
        
        return {
            "correctness": correctness_result.score,
            "conciseness": conciseness_result.score, 
            "code_checker": code_checker_result.score if code_checker_result else None,
            "overall_score": overall_score,
            "correctness_reasoning": correctness_result.reasoning,
            "conciseness_reasoning": conciseness_result.reasoning,
            "code_reasoning": code_checker_result.reasoning if code_checker_result else "No code to evaluate",
            "evaluation_text": f"Оценка качества ответа: {overall_score} из 5" if overall_score is not None else "Ошибка при оценке качества ответа"
        }
    
    def _evaluate_correctness(self, user_query: str, answer: str, execution_result: str, answer_reasoning: str = "") -> EvaluationResult:
        """Оценка корректности ответа"""
        correctness_prompt = """
        You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology

  When scoring, you should penalize:
  - Factual errors or inaccuracies
  - Incomplete or partial answers
  - Misleading or ambiguous statements
  - Incorrect terminology
  - Logical inconsistencies
  - Missing key information
</Rubric>

<Instructions>
  - Carefully read the input, intermediate reasoning and output
  - Check for factual accuracy and completeness
  - Focus on correctness of information rather than style or verbosity
</Instructions>

<Reminder>
  The goal is to evaluate factual correctness and completeness of the response.
</Reminder>

<input>
{{inputs}}
</input>

<output>
{{outputs}}
</output>
        """
        
        # Заменяем плейсхолдеры реальными данными
        inputs = user_query
        outputs = f"Ответ: {answer}\nРезультат выполнения: {execution_result}"
        if answer_reasoning:
            outputs += f"\nЛогика формирования ответа: {answer_reasoning}"
        
        prompt_with_data = correctness_prompt.replace("{{inputs}}", inputs).replace("{{outputs}}", outputs)
        
        messages = [
            SystemMessage(content=prompt_with_data),
            HumanMessage(content="Оцени от 1 до 5:")
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt_with_data},
                    {"role": "user", "content": "Оцени от 1 до 5 и предоставь reasoning:"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation_result",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["score", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return EvaluationResult(result["score"], result["reasoning"])
        except Exception:
            logger.exception("Error evaluating correctness")
            return EvaluationResult(None, "Error occurred during evaluation")
    
    def _evaluate_conciseness(self, user_query: str, answer: str, answer_reasoning: str = "") -> EvaluationResult:
        """Оценка краткости ответа"""
        conciseness_prompt = """
You are an expert data labeler evaluating model outputs for conciseness. Your task is to assign a score based on the following rubric:

<Rubric>
  A perfectly concise answer:
  - Contains only the exact information requested.
  - Uses the minimum number of words necessary to convey the complete answer.
  - Omits pleasantries, hedging language, and unnecessary context.
  - Excludes meta-commentary about the answer or the model's capabilities.
  - Avoids redundant information or restatements.
  - Does not include explanations unless explicitly requested.

  When scoring, you should deduct points for:
  - Introductory phrases like "I believe," "I think," or "The answer is."
  - Hedging language like "probably," "likely," or "as far as I know."
  - Unnecessary context or background information.
  - Explanations when not requested.
  - Follow-up questions or offers for more information.
  - Redundant information or restatements.
  - Polite phrases like "hope this helps" or "let me know if you need anything else."
</Rubric>

<Instructions>
  - Carefully read the input and output.
  - Check for any unnecessary elements, particularly those mentioned in the <Rubric> above.
  - The score should reflect how close the response comes to containing only the essential information requested based on the rubric above.
</Instructions>

<Reminder>
  The goal is to reward responses that provide complete answers with absolutely no extraneous information.
</Reminder>

<input>
{{inputs}}
</input>

<output>
{{outputs}}
</output>
        """
        
        # Заменяем плейсхолдеры реальными данными
        inputs = user_query
        outputs = answer

        
        prompt_with_data = conciseness_prompt.replace("{{inputs}}", inputs).replace("{{outputs}}", outputs)
        
        messages = [
            SystemMessage(content=prompt_with_data),
            HumanMessage(content="Оцени от 1 до 5:")
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt_with_data},
                    {"role": "user", "content": "Оцени от 1 до 5 и предоставь reasoning:"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation_result",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["score", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return EvaluationResult(result["score"], result["reasoning"])
        except Exception:
            logger.exception("Error evaluating conciseness")
            return EvaluationResult(None, "Error occurred during evaluation")
    
    def _evaluate_code_quality(self, pandas_code: str, user_query: str, code_reasoning: str = "") -> EvaluationResult:
        """Оценка качества сгенерированного кода"""
        code_checker_prompt = """
        You are an expert code reviewer evaluating code for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct code solution:
  - Solves the problem completely as specified in the input
  - Handles all edge cases appropriately
  - Contains absolutely no bugs or logical errors
  - Uses efficient and appropriate algorithms/data structures
  - Follows language-specific best practices
  - Has correct syntax and would compile/run without errors

  When scoring, you should penalize:
  - Logical errors or bugs that would cause incorrect behavior
  - Missing edge case handling
  - Overly inefficient implementations when better approaches exist
  - Incomplete solutions that don't address all requirements
  - Syntax errors that would prevent compilation/execution
  - Security vulnerabilities or unsafe practices
</Rubric>

<Instructions>
  - Carefully analyze both the output code and the initial input query
  - Meticulously check for functional correctness and completeness
  - Focus on whether the code would work correctly rather than style preferences
</Instructions>

<Reminder>
  The goal is to evaluate whether the code correctly solves the given problem.
</Reminder>

<input>
{{inputs}}
</input>

<output>
{{outputs}}
</output>

        """
        
        # Заменяем плейсхолдеры реальными данными
        inputs = user_query
        outputs = f"Код:\n{pandas_code}"
        if code_reasoning:
            outputs += f"\nЛогика генерации кода: {code_reasoning}"
        
        prompt_with_data = code_checker_prompt.replace("{{inputs}}", inputs).replace("{{outputs}}", outputs)
        
        messages = [
            SystemMessage(content=prompt_with_data),
            HumanMessage(content="Оцени от 1 до 5:")
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt_with_data},
                    {"role": "user", "content": "Оцени от 1 до 5 и предоставь reasoning:"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation_result",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["score", "reasoning"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return EvaluationResult(result["score"], result["reasoning"])
        except Exception:
            logger.exception("Error evaluating code quality")
            return EvaluationResult(None, "Error occurred during evaluation")
