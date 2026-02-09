from rag.vectorstore import DOCUMENTS
from rag.llm import run_llm

def keyword_retrieve(query, k=2):
    query_words = set(query.lower().split())

    scored = []
    for text, meta in DOCUMENTS:
        text_words = set(text.lower().split())
        score = len(query_words & text_words)
        if score > 0:
            scored.append((score, text, meta))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]

def rag(query: str):
    retrieved = keyword_retrieve(query)

    if not retrieved:
        return "The answer is not mentioned in the document.", []

    context = "\n\n".join([x[1] for x in retrieved])

    prompt = f"""
You are a strict document-based assistant.

CONTEXT:
{context}

QUESTION:
{query}

RULES:
- Answer ONLY from the context
- If not found, say it is not mentioned
"""

    answer = run_llm(prompt)

    sources = [
        {
            "page": meta["page"],
            "chunk_id": meta["chunk_id"]
        }
        for _, _, meta in retrieved
    ]

    return answer, sources
