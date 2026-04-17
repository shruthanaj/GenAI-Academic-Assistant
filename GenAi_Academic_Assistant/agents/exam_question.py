"""Exam Question Agent — generates exam-style questions and answers."""
from __future__ import annotations
from backend.llm_client import call_llm

SYSTEM = """You are an expert exam paper setter for engineering students.
Generate exam-style questions based on the topic or provided study material.
Include a mix of:
- 2-mark short answer questions
- 5-mark descriptive questions
- MCQs with 4 options and correct answer marked
Always provide model answers after each question.
Format nicely with markdown.
IMPORTANT: Generate COMPLETE questions. Do not truncate or stop mid-question."""


class ExamQuestionAgent:
    def answer(self, question: str, context: str = "") -> str:
        # Check if user is asking for many questions
        question_lower = question.lower()
        is_large_request = any(
            keyword in question_lower
            for keyword in ["50", "100", "many", "lot", "comprehensive", "full", "complete"]
        )

        # Allocate tokens based on request size
        if is_large_request:
            max_tokens = 8000  # Maximum for large requests
        else:
            max_tokens = 6000  # Standard allocation

        ctx_block = f"\n\n**Study Material / Context:**\n{context}" if context else ""
        user_msg = f"Generate exam questions on: {question}{ctx_block}"

        response = call_llm(SYSTEM, user_msg, max_tokens=max_tokens)

        # Add warning if response might be truncated
        if is_large_request and len(response) > 7500:
            response += "\n\n⚠️ **Note**: Response was optimized for token limit. For more questions, try asking for specific topics or question types separately."

        return response
