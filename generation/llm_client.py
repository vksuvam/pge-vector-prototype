"""
llm_client.py
Handles all Groq API calls. Supports model switching per request.
Tracks token usage for metadata.
"""

from typing import Dict, Any
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODELS, DEFAULT_MODEL


def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Export it as an environment variable:\n"
            "  export GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=GROQ_API_KEY)


def validate_model(model: str) -> str:
    """
    Validates the requested model against supported Groq models.
    Falls back to default if not recognized.
    """
    if model not in GROQ_MODELS:
        print(f"[LLM] Unknown model '{model}', falling back to '{DEFAULT_MODEL}'")
        return DEFAULT_MODEL
    return model


def generate(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float = 0.1,   # low temp for factual/technical docs
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Sends a chat completion request to Groq.

    Returns:
    {
        "answer":        str,
        "input_tokens":  int,
        "output_tokens": int,
        "total_tokens":  int,
        "model_used":    str,
    }
    """
    model = validate_model(model)
    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    usage = response.usage
    return {
        "answer": response.choices[0].message.content.strip(),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "model_used": model,
    }
