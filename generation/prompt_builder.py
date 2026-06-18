"""
prompt_builder.py

Two responsibilities:
1. Detect whether the query involves a process/steps/procedure.
2. Build the appropriate system + user prompt accordingly.

PROCESS RULE (per requirements):
    If the query — directly OR indirectly — involves a procedure, steps, or process,
    return the FULL verbatim content from the source. Do NOT summarize.
    This applies to both Greenbook and Tariff documents.

    Indirect detection: "how do I apply for new service?" has no word 'steps' but
    implies a process. We catch this via keyword matching + an LLM-side instruction.
"""

from typing import List, Dict, Any
from config import PROCESS_KEYWORDS


def is_process_query(query: str) -> bool:
    """
    Returns True if the query likely involves a procedure or step-by-step process.
    Matching is case-insensitive and checks for any keyword/phrase from config.
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in PROCESS_KEYWORDS)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a numbered context block for the prompt.
    Each chunk is tagged with its document source and page number.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_label = chunk["doc_name"].replace("_", " ").title()
        parts.append(
            f"[Context {i} | Source: {doc_label} | Page: {chunk['page_no']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


# SYSTEM_PROMPT_GENERAL = """You are an expert assistant for PG&E's Service Planning & Design (SP&D) department.
# You answer questions using ONLY the provided context from the PG&E Greenbook and Tariff documents.

# Rules:
# - Answer based strictly on the provided context. Do not use outside knowledge.
# - If the context does not contain enough information to answer, say: "I could not find sufficient information in the PG&E documents to answer this question."
# - Always cite the source document and page number when referencing specific information.
# - Be clear, concise, and professional.
# - Do not speculate or make assumptions beyond what the documents state."""


# SYSTEM_PROMPT_PROCESS = """You are an expert assistant for PG&E's Service Planning & Design (SP&D) department.
# You answer questions using ONLY the provided context from the PG&E Greenbook and Tariff documents.

# IMPORTANT — PROCESS/STEPS RULE:
# This query involves a procedure, process, or set of steps.
# You MUST return the complete process EXACTLY as written in the source document.
# - Do NOT summarize, condense, or paraphrase any steps.
# - Do NOT skip or merge any steps.
# - Reproduce the full process verbatim from the context provided.
# - If multiple context chunks contain parts of the same process, combine them in order.
# - After the verbatim content, you may add a brief note about the source, but the steps themselves must be unmodified.
# - If the context does not contain a process or steps relevant to the query, say so clearly."""


SYSTEM_PROMPT_GENERAL = """You are an expert assistant for PG&E's Service Planning & Design (SP&D) department.
You answer questions using ONLY the provided context from the PG&E Greenbook and Tariff documents.

IMPORTANT CITATION RULES:
- Use numbered citations [1], [2], [3], etc. in your answer
- Each citation [N] refers to the source listed in the sources section below
- Do NOT include source titles, page numbers, or quotes in your answer
- Do NOT say "According to...", "See page...", "Table...", "Figure..."
- Let the numbered citations and sources section handle all references
- Keep your answer focused on the information, not on where it comes from

Example:
BAD:  "According to the Greenbook page 45, the clearance is 10 feet."
GOOD: "The clearance is 10 feet [1]."

Rules:
- Answer based strictly on the provided context
- If context is insufficient, say: "I could not find sufficient information..."
- Keep answers clean and readable
- Let citations and sources section provide full reference details
"""


SYSTEM_PROMPT_PROCESS = """You are an expert assistant for PG&E's Service Planning & Design (SP&D) department.

PROCESS/STEPS RULE:
This query asks for a procedure, process, or set of steps.
You MUST return the complete process EXACTLY as written in the source document.
- Do NOT summarize, condense, or paraphrase any steps
- Do NOT skip or merge any steps
- Reproduce the full process verbatim
- Use [1], [2], [3] citations to mark which source each step comes from
- Do NOT add phrases like "According to...", "See page...", "As mentioned in..."

Example format:
1. Select the mandrel [1]
2. Pull through the conduit [1]
3. Attach the pulling tape [2]

Then after the steps, you can add a brief note:
NOTE: Steps 1-3 from Greenbook page 45, Step 4 from Tariff document page 12.
"""

def build_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
    has_image: bool = False,
) -> tuple[str, str]:
    """Build system + user prompts with citation instruction."""
    
    # Build numbered context with labels
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        doc_label = chunk["doc_name"].replace("_", " ").replace("-", " ").title()
        context_lines.append(
            f"[{i}] Source: {doc_label}, Page {chunk['page_no']}\n"
            f"Content: {chunk['text']}"
        )
    context = "\n\n".join(context_lines)

    if is_process_query(query):
        system_prompt = SYSTEM_PROMPT_PROCESS
        user_message = f"""Context Sources:
{context}

---

Question: {query}

REMEMBER: Use citations [1], [2], etc. to mark which source each step comes from. 
Do NOT include source titles or page numbers in your answer - that's what [1], [2] are for."""

    else:
        system_prompt = SYSTEM_PROMPT_GENERAL
        user_message = f"""Context Sources:
{context}

---

Question: {query}

REMEMBER: Use numbered citations [1], [2], [3], etc. to reference sources.
Do NOT say "According to", "See page", "Figure", "Table", etc.
Keep your answer clean - citations handle all references."""

    return system_prompt, user_message
