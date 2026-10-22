"""MedRAC-QA: a lightweight LLM + RAG + CAG framework for medical health question answering.

Modules, in the order a question flows through them:

    safety          -> emergency screening and "never cache" rules
    query_analysis  -> intent, clinical qualifiers and focus terms of a question
    embedder        -> all-MiniLM-L6-v2 sentence embeddings (ONNX, CPU)
    cache           -> CAG: safety-constrained semantic answer cache
    knowledge       -> RAG: vector search over the MedQuAD knowledge base
    evidence        -> evidence sufficiency gate and citation validation
    llm             -> OpenAI-compatible chat client (Groq, Ollama, ...)
    pipeline        -> puts all of the above together
"""

__version__ = "1.0.0"
