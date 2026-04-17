"""Study Planner Agent — creates personalized revision schedules based on student assessment."""
from __future__ import annotations
from backend.llm_client import call_llm

# (ASSESSMENT_SYSTEM removed — assessment is now handled via a UI form, not LLM)

# System prompt for plan generation
PLANNER_SYSTEM = """You are an expert academic study planner for engineering students.
Create PERSONALIZED, realistic study/revision plans based on the student's assessment.

Generate a plan that:
- Focuses on WEAK topics (give 50% more time there)
- Reviews STRONG topics briefly
- Matches student's learning stage
- Fits their available time
- Includes checkpoints to assess progress

Include:
- Day-by-day schedule in table format
- Time allocation per topic
- Revision tips tailored to their level
- Last-minute exam strategies
Use markdown tables for clarity."""


class StudyPlannerAgent:
    def get_assessment_options(self) -> dict:
        """
        Returns structured options for assessment questions.
        Used by frontend to render choice-based widgets (radio, selectbox).
        """
        return {
            "q1_level": ["Beginner", "Intermediate", "Advanced"],
            "q3_time": ["< 1 week", "1-2 weeks", "2-3 weeks", "3+ weeks"],
            "q4_style": ["Visual (diagrams/charts)", "Textual (reading/notes)", "Practical (problems)", "Mixed"],
        }

    def ask_assessment_questions(self, context: str = "") -> str:
        """
        Step 1: Return a static prompt telling the user to fill the assessment form.
        (No LLM call — the form widget in the frontend collects the answers.)
        """
        topics_hint = ""
        if context:
            import re
            # Strip raw file/slide markers like === filename === or === Slide 1 ===
            clean = re.sub(r"={2,}[^=]+={2,}", " ", context)
            # Collapse multiple spaces/newlines into a single space
            clean = re.sub(r"\s+", " ", clean).strip()
            # Take first 130 meaningful characters
            snippet = clean[:130] + ("…" if len(clean) > 130 else "")
            topics_hint = f"\n\n📚 **Detected from your material:** _{snippet}_"

        return (
            "📋 **Let's personalize your study plan!**\n\n"
            "Please fill out the quick assessment form below 👇 "
            "so I can create a plan that fits **your** level, schedule, and learning style."
            f"{topics_hint}"
        )

    def create_personalized_plan(self,
                                 original_question: str,
                                 student_assessment: str,
                                 context: str = "") -> str:
        """
        Step 2: Create personalized plan based on student's answers

        Args:
            original_question: The initial "create plan for X" question
            student_assessment: Student's answers from assessment form
            context: Material from uploaded documents
        """

        ctx_block = f"\n\n**Topics Available in Your Material:**\n{context[:1000]}" if context else ""

        assessment_block = f"""
**Student's Learning Profile:**
{student_assessment}

**Their Request:** {original_question}
"""

        user_msg = f"""{assessment_block}

Create a PERSONALIZED study plan based on their profile that:
- Allocates MORE time (50%) to weak topics
- Less time to strong topics (just quick reviews)
- Matches their {'beginner' if 'beginner' in student_assessment.lower() else 'advanced' if 'advanced' in student_assessment.lower() else 'intermediate'} level
- Fits their available time
- Includes daily milestones{ctx_block}"""

        return call_llm(PLANNER_SYSTEM, user_msg, max_tokens=3500)

    def answer(self, question: str, context: str = "") -> str:
        """
        Main answer method - asks assessment questions
        (Actual plan generation happens in frontend after form submission)
        """
        return self.ask_assessment_questions(context)
