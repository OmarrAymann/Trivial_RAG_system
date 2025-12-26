import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import time
from contextlib import asynccontextmanager
import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app import config
from app.rag import RagPipeline
from app.utils import logger


pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting RAG application")
    try:
        pipeline = RagPipeline()
        pipeline.load_resources()
        logger.info("=" * 60)
        logger.info("RAG System is running :")
        logger.info(f"API: http://0.0.0.0:{config.api_port}")
        logger.info(f"UI:  http://localhost:{config.api_port}/ui")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    yield
    logger.info("Shutting down RAG application")

app = FastAPI(
    title="RAG System",
    description="Production RAG system with TriviaQA dataset",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: list[str]
    latency_ms: int

@app.get("/")
async def root():
    return {
        "message": "RAG System API",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "healthy",
        "index_size": pipeline.index.ntotal,
        "metadata_count": len(pipeline.metadata)
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    start_time = time.perf_counter()
    try:
        result = pipeline.query(request.question, request.top_k)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            retrieved_context=result['retrieved_context'],
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def gradio_query(question: str, top_k: int = 5):
    if not question or not question.strip():
        return "Please enter a question", "", "", 0
    start_time = time.perf_counter()
    try:
        result = pipeline.query(question.strip(), top_k)
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)
        
        if result.get('error'):
            return result['error'], "", "", 0
        
        answer = result['answer']
        context = "\n\n---\n\n".join([
            f"[Chunk {i+1}]\n{chunk}" 
            for i, chunk in enumerate(result['retrieved_context'])
        ])
        metadata = f"Retrieved {len(result['retrieved_context'])} chunks | Latency: {latency_ms}ms"
        return answer, context, metadata, latency_ms
        
    except Exception as e:
        logger.error(f"Gradio query failed: {e}")
        return f"Error: {str(e)}", "", "", 0

def create_gradio_interface():
    with gr.Blocks(title="RAG System") as demo:
        gr.Markdown("RAG System")
        gr.Markdown("Ask questions based on the TriviaQA dataset.")
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter your question here : ",
                    lines=3
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of context chunks"
                )
                submit_btn = gr.Button("Submit", variant="primary")
                
            with gr.Column(scale=3):
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=5,
                    interactive=False
                )
                metadata_output = gr.Textbox(
                    label="Info",
                    lines=1,
                    interactive=False
                )
        
        with gr.Row():
            context_output = gr.Textbox(
                label="Retrieved Context",
                lines=10,
                interactive=False
            )
        
        submit_btn.click(
            fn=gradio_query,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, context_output, metadata_output, gr.Number(visible=False)]
        )
    return demo

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

if __name__ == "__main__":
    
    uvicorn.run(
        "app.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level="error" 
    )