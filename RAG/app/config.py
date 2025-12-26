import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data"
index_dir = base_dir / "faiss_index"

# embedding settings
embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
top_k = int(os.getenv("TOP_K", "3"))

# LLM settings
ollama_model = os.getenv("OLLAMA_MODEL", "llama2:7b-chat-q4_0")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "150"))

# dataset settings
dataset_name = os.getenv("DATASET_NAME", "trivia_qa")
dataset_split = os.getenv("DATASET_SPLIT", "train")
max_documents = int(os.getenv("MAX_DOCUMENTS", "2000"))

# api settings
api_host = os.getenv("API_HOST", "0.0.0.0")
api_port = int(os.getenv("API_PORT", "8000"))

# gradio settings
gradio_host = os.getenv("GRADIO_HOST", "0.0.0.0")
gradio_port = int(os.getenv("GRADIO_PORT", "7860"))