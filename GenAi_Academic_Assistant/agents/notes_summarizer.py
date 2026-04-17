"""Notes Summarizer Agent — summarises PDFs / chapters into key points."""
from __future__ import annotations
from backend.llm_client import call_llm

SYSTEM = """You are an academic notes summarizer for engineering students.
Create concise, well-structured summaries of study material.
Format the summary as:
- **Key Concepts** (bullet list)
- **Important Definitions**
- **Quick Revision Points**
- **Remember for Exams** (critical facts)
Be exam-focused. Use markdown formatting."""


class NotesSummarizerAgent:
    def answer(self, question: str, context: str = "") -> str:
        if context:
            user_msg = f"Summarize the following material related to '{question}':\n\n{context}"
        else:
            user_msg = f"Provide a detailed summary on the topic: {question}"
        return call_llm(SYSTEM, user_msg, max_tokens=3000)
