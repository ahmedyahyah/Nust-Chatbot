"""
RAG Chain - Groq LLM + Tavily web search fallback.
"""

import os
import requests
from typing import List, Tuple
from groq import Groq


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")
    return Groq(api_key=api_key)


def search_web(query: str) -> str:
    """Search web using Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": f"NUST Pakistan {query}",
                "search_depth": "basic",
                "max_results": 4,
                "include_answer": True,
            },
            timeout=10,
        )
        data = response.json()
        results = []
        if data.get("answer"):
            results.append(f"Summary: {data['answer']}")
        for r in data.get("results", [])[:3]:
            content = r.get("content", "")[:400]
            title = r.get("title", "")
            if content:
                results.append(f"Source: {title}\n{content}")
        return "\n\n".join(results)
    except Exception:
        return ""


def generate_answer(
    question: str,
    context_chunks: List[str],
    chat_history: List[dict],
) -> Tuple[str, str]:
    """
    Generate answer using RAG + web fallback.

    Returns:
        (answer, source) — source is 'document', 'web', or 'none'
    """
    client = get_groq_client()

    doc_context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
    source = "document"

    # Always search web as well for better answers
    web_context = search_web(question)

    # Determine source label
    if not context_chunks or len(doc_context.strip()) < 200:
        source = "web" if web_context else "none"
    else:
        source = "document"

    system_prompt = """You are the official NUST (National University of Sciences and Technology) AI Assistant.
You ONLY answer questions related to NUST Pakistan.

RULES:
1. If question is about NUST → answer using documents and web search
2. If question is NOT about NUST → politely say: "I can only help with NUST-related questions. Please ask me about admissions, programs, fees, scholarships, or campus life."
3. Never answer general knowledge questions unrelated to NUST
4. Never make up fees, dates, or admission criteria
5. Be friendly, concise, and use bullet points for lists"""

    context_parts = []
    if doc_context:
        context_parts.append(f"=== NUST DOCUMENTS ===\n{doc_context}")
    if web_context:
        context_parts.append(f"=== WEB SEARCH ===\n{web_context}")

    context_message = "\n\n".join(context_parts) if context_parts else "No context available."

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-8:]:
        messages.append(msg)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context_message}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content, source