import time
import logging
from typing import List,Dict,Tuple,Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return result, latency_ms
    return wrapper


def clean_text(text: str, lowercase: bool = False) -> str:
    if not text:
        return ""
    text = text.strip()
    text = " ".join(text.split())
    if lowercase:
        text = text.lower()
    return text


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def validate_query(query: str) -> Tuple[bool, str]:
    if not query or not query.strip():
        return False, "Query cannot be empty"

    query_len = len(query.strip())
    if query_len < 5:
        return False, "Query is too short (minimum 5 characters)"

    if query_len > 1000:
        return False, "Query too long (maximum 1000 characters)"

    return True, ""


def format_context(chunks: List[Dict[str, Any]]) -> List[str]:
    return [
        clean_text(chunk.get("text", ""))
        for chunk in chunks
        if chunk.get("text")
    ]


def build_prompt(question: str, context: List[str], fallback: str = "I don't have enough information to answer this question.") -> str:
    if not context:
        return f"Question: {question}\n\nAnswer: {fallback}"

    context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
    prompt = f"""You are a helpful assistant. Answer the question based on the context provided below. If the answer is not in the context, respond with: "{fallback}"

Context:
{context_text}

Question: {question}

Answer:"""
    return prompt