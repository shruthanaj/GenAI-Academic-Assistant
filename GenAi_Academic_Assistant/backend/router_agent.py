"""Router Agent — classifies queries and routes to appropriate specialist agent."""
from __future__ import annotations
from agents.concept_explainer import ConceptExplainerAgent
from agents.exam_question import ExamQuestionAgent
from agents.notes_summarizer import NotesSummarizerAgent
from agents.study_planner import StudyPlannerAgent


class RouterAgent:
    """Routes user questions to the most appropriate specialist agent."""

    def __init__(self):
        self.concept_agent = ConceptExplainerAgent()
        self.exam_agent = ExamQuestionAgent()
        self.summary_agent = NotesSummarizerAgent()
        self.planner_agent = StudyPlannerAgent()

    def route_and_answer(self, question: str, context: str = "") -> tuple[str, str]:
        """
        Route question to appropriate agent and return (agent_name, answer).

        Args:
            question: User's question
            context: Retrieved context from RAG pipeline

        Returns:
            Tuple of (agent_name, answer_text)
        """
        question_lower = question.lower()

        # ── Priority 1: Study Planner ─────────────────────────────────────────
        # "study" alone is too broad — only match clear planning-intent words
        if any(word in question_lower for word in [
            "plan", "schedule", "revision", "timetable",
            "prepare", "day by day", "weekly plan", "study plan",
            "planner", "days left"
        ]):
            agent_name = "Study Planner Agent"
            answer = self.planner_agent.answer(question, context)

        # ── Priority 2: Exam / Q&A Generator ─────────────────────────────────
        elif any(word in question_lower for word in [
            "question", "quiz", "exam", "test", "practice",
            "q&a", "qa", "mcq", "generate q", "questionnaire",
            "mock", "generate questions", "answer key"
        ]):
            agent_name = "Exam Question Agent"
            answer = self.exam_agent.answer(question, context)

        # ── Priority 3: Notes Summarizer ──────────────────────────────────────
        elif any(word in question_lower for word in [
            "summary", "summarize", "summarise", "chapter",
            "notes", "recap", "overview", "key points", "brief"
        ]):
            agent_name = "Notes Summarizer Agent"
            answer = self.summary_agent.answer(question, context)

        # ── Default: Concept Explainer ────────────────────────────────────────
        else:
            agent_name = "Concept Explainer Agent"
            answer = self.concept_agent.answer(question, context)

        return agent_name, answer
