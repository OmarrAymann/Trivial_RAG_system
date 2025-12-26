# Trivial_RAG_system

The system processes the TriviaQA dataset to answer questions with relevant context retrieval and generation.


## Project Structure
```
rag/
├── app/
│   ├── config.py       # Configuration management
│   ├── utils.py        # Utility functions
│   ├── ingest.py       # Data ingestion pipeline
│   ├── rag.py          # RAG pipeline implementation
│   └── main.py         # FastAPI application
├── data/               # Dataset storage (gitignored)
├── faiss_index/        # Persisted FAISS index (gitignored)
├── evaluation/         # Evaluation results
├── requirements.txt    # Python dependencies
├── dockerfile          # Container definition
└── .env               # Environment configuration
```
## Architecture Overview
### Pipeline Flow
```
User Query
    |
    v
Query Embedding (sentence-transformers)
    |
    v
Vector Search (FAISS L2 distance)
    |
    v
Top-K Context Retrieval
    |
    v
Prompt Construction
    |
    v
LLM Generation (Ollama LLaMA 2 7B)
    |
    v
Final Answer + Retrieved Context
```

### Components

**Ingestion Pipeline** (`process.py`) 
- Loads TriviaQA dataset from Hugging Face
- Cleans and preprocesses text
- Chunks documents with configurable size and overlap
- Generates dense embeddings using sentence-transformers
- Builds FAISS index for efficient similarity search
- Persists index and metadata to disk

**Retrieval Pipeline** (`rag.py`)
- Embeds incoming queries
- Performs vector similarity search via FAISS
- Retrieves top-K most relevant document chunks
- Returns scored results with metadata

**Generation Pipeline** (`rag.py`)
- Formats retrieved context into structured prompt
- Sends prompt to Ollama local LLM
- Returns generated answer with source attribution

**API Layer** (`main.py`)
- FastAPI REST endpoints for programmatic access
- Gradio web interface for interactive queries
- Health checks and monitoring endpoints
- Request validation and error handling

## Key Design Decisions

### Chunking Strategy

**Configuration**: 512 tokens per chunk with 50 token overlap

**Rationale**:
- 512 tokens balances context completeness with retrieval precision
- 50 token overlap prevents information loss at chunk boundaries
- Token-based chunking (approximated by words) ensures consistent embedding input sizes
- Overlap allows questions spanning chunk boundaries to retrieve relevant information

### Embedding Model

**Model**: sentence-transformers/all-MiniLM-L6-v2

**Rationale**:
- Produces about 384-dimensional embeddings with strong semantic capture
- Fast inference suitable for real-time query embedding
- Pre-trained on semantic similarity tasks
- Compact model size enables local deployment

### Vector Store

**Technology**: FAISS with L2 distance

**Rationale**:
- Highly optimized C++ implementation with Python bindings
- IndexFlatL2 provides exact nearest neighbor search
- No external service dependencies
- Suitable for dataset size (2000 documents,2000 vectors)
- Simple persistence via index serialization

### LLM Selection

**Model**: Ollama LLaMA 2 7B (quantized)

**Rationale**:
- Runs entirely locally without external API dependencies
- Zero per-query cost
- Complete data privacy
- 7B parameter size balances quality with inference speed
- Quantization reduces memory footprint

**Performance Characteristics**:
- First query: 3-4 seconds (model loading)
- Subsequent queries: 2-2.5 seconds
- Acceptable latency for interactive applications

## Local Setup

### Prerequisites

- Python 3.11 or higher
- 8GB RAM minimum
- Ollama installed locally
- 5GB free disk space

### Installation

1. Install Ollama and download model:
```bash
brew install ollama
ollama pull llama2:7b
ollama serve
```

2. Clone repository and install dependencies:
```bash
git clone 
cd rag
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env if needed (defaults are suitable for local development)
```

4. Run data ingestion:
```bash
python -m app.process
```

Expected output:
- Downloads TriviaQA dataset from Hugging Face
- Processes 2000 documents
- Generates embeddings
- Builds and persists FAISS index
- Duration: 1-2 minutes

5. Start application:
```bash
python -m app.main
```

Access points:
- Web UI: http://localhost:8000/ui
- REST API: http://localhost:8000
- 
### API Usage

Query endpoint:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "top_k": 5
  }'
```

Response format:
```json
{
  "question": "What is the capital of France?",
  "answer": "Paris",
  "retrieved_context": [
    "Context chunk 1...",
    "Context chunk 2..."
  ],
  "latency_ms": 1284
}
```

## Docker Deployment

### Build Image
```bash
docker build -t rag-system .
```

### Run Container

Connect to host Ollama instance:
```bash
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-system
```

Linux with host networking:
```bash
docker run --network=host rag-system
```

### Notes

- Container runs ingestion on startup
- Ollama must be accessible from container
- First startup takes 3-5 minutes for ingestion
- Subsequent restarts use cached index

## System Evaluation

### Test Configuration

- Dataset: TriviaQA (2000 documents)
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- LLM: Ollama LLaMA 2 7B (quantized)
- Top-K: 5 chunks
- Chunk Size: 512 tokens
- Chunk Overlap: 50 tokens

### Performance Summary

**Retrieval Accuracy**
- Context contains answer: (6/8)
- Context partially contains answer: (2/8)
- No relevant context found: 0%

**Latency:**

- Average: 2284ms
- Range: 1767ms - 4034ms
- Note: Latency includes retrieval (approximately 100ms) and LLM generation (approximately 1700ms).

### Analysis

**Strengths**:
- High retrieval accuracy indicates effective embedding and chunking strategy
- Answer quality demonstrates successful context utilization by LLM
- Consistent latency suitable for interactive applications
- Zero operational cost enables unlimited queries

**Limitations**:
- Latency higher than cloud-based APIs (trade-off for local inference)
- Small dataset limits answer coverage
- No conversation history or multi-turn dialogue

## Configuration

Key parameters in `.env`:
```bash
# Embedding configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Retrieval configuration
TOP_K=5

# LLM configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:7b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512

# Dataset configuration
MAX_DOCUMENTS=2000
```

