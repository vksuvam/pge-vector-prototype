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


SYSTEM_PROMPT_GENERAL = """You are an expert assistant for PG&E's Service Planning & Design (SP&D) department.
You answer questions using ONLY the provided context from the PG&E Greenbook and Tariff documents.

Rules:
- Answer based strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say: "I could not find sufficient information in the PG&E documents to answer this question."
- Always cite the source document and page number when referencing specific information.
- Be clear, concise, and professional.
- Do not speculate or make assumptions beyond what the documents state."""


SYSTEM_PROMPT_PROCESS = """You are an expert assistant for PG&E's Service Planning & Design (SP&D) department.
You answer questions using ONLY the provided context from the PG&E Greenbook and Tariff documents.

IMPORTANT — PROCESS/STEPS RULE:
This query involves a procedure, process, or set of steps.
You MUST return the complete process EXACTLY as written in the source document.
- Do NOT summarize, condense, or paraphrase any steps.
- Do NOT skip or merge any steps.
- Reproduce the full process verbatim from the context provided.
- If multiple context chunks contain parts of the same process, combine them in order.
- After the verbatim content, you may add a brief note about the source, but the steps themselves must be unmodified.
- If the context does not contain a process or steps relevant to the query, say so clearly."""


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> tuple[str, str]:
    """
    Builds (system_prompt, user_message) tuple for the LLM.

    Returns different system prompts depending on whether the query
    is detected as a process/steps query.
    """
    context = _format_context(chunks)
    process_query = is_process_query(query)

    system_prompt = SYSTEM_PROMPT_PROCESS if process_query else SYSTEM_PROMPT_GENERAL

    user_message = f"""Context from PG&E Documents:
{context}

---
Question: {query}

{"REMINDER: This question involves a process or steps. Return the FULL verbatim content. Do not summarize." if process_query else ""}"""

    return system_prompt, user_message
