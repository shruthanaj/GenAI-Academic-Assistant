"""Concept Explainer Agent — explains topics in beginner-friendly terms."""
from __future__ import annotations
from backend.llm_client import call_llm

# Words that are "instructional/command" only — not real topic names
_COMMAND_ONLY_WORDS = {
    "explain", "in", "simple", "simply", "terms", "words", "easy",
    "what", "is", "are", "tell", "me", "about", "describe", "elaborate",
    "clarify", "give", "understanding", "understand", "briefly", "detail",
    "detailed", "short", "clearly", "layman", "basic", "basics",
    "a", "an", "the", "this", "that", "it", "how", "does", "do",
    "please", "can", "you", "just", "more", "better", "again",
}

SYSTEM = """You are a friendly academic tutor helping an engineering student.

CRITICAL RULE: You MUST prioritize the provided study material above general knowledge.
- If study material is provided, your answer MUST be based on it first
- If a term appears in the material, use THAT specific context/definition
- Only use general knowledge if NOT covered in the material
- Always cite which part of the material you're using
- If the material doesn't contain the answer, say: "This topic isn't covered in your uploaded material"

Explain clearly using:
- Analogies and real-world examples
- Bullet points for clarity
- Step-by-step breakdowns
Format nicely with markdown."""


def _is_vague_query(question: str) -> bool:
    """Return True if the question has no real topic — only command words."""
    words = set(question.lower().replace("?", "").replace("!", "").split())
    meaningful = words - _COMMAND_ONLY_WORDS
    return len(meaningful) == 0


class ConceptExplainerAgent:
    def answer(self, question: str, context: str = "") -> str:

        # ── Guard: vague query with no topic ──────────────────────────────────
        if _is_vague_query(question):
            return (
                "🤔 **Which topic would you like me to explain?**\n\n"
                "Please mention a specific concept — for example:\n"
                "- *\"Explain Lexical Analysis in simple terms\"*\n"
                "- *\"What is a Symbol Table?\"*\n"
                "- *\"Explain Scope in compiler design\"*\n\n"
                "I'll use your uploaded study material to give you a clear, simple explanation!"
            )

        # ── Guard: no study material uploaded ────────────────────────────────
        if not context:
            return (
                "📂 **Please upload study material first!**\n\n"
                "I work best with your lecture notes, textbooks, or PDFs. "
                "Once uploaded, I can explain any concept from your material in simple terms."
            )

        # ── Normal explanation using RAG context ──────────────────────────────
        ctx_block = f"""
**IMPORTANT - BASE YOUR ANSWER ON THIS MATERIAL ONLY:**
```
{context}
```

Remember: Use the above material to answer "{question}". If "{question}" isn't in the material, say so."""

        user_msg = f"Explain this concept based on the study material provided above:\n{question}"
        return call_llm(SYSTEM, user_msg + ctx_block, max_tokens=2000)

