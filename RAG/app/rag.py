import pickle
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app import config
from app.utils import logger, validate_query, format_context, build_prompt

class RagPipeline:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self.llm_client = None
        
    def load_resources(self):
        self.embedding_model = SentenceTransformer(config.embedding_model_name)
        index_path = config.index_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found, run ingest.py file first")
        
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        metadata_path = config.index_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found, run ingest.py file first")
        
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Resources loaded: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries")
        logger.info("Ollama LLM configured")
    
    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.embedding_model.encode([query])
        return embedding.astype('float32')
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        
        results = []
        if top_k is None:
            top_k = config.top_k

        query_embedding = self.embed_query(query)
        distances, indices = self.index.search(query_embedding, top_k)

        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(distance)
                result['score'] = float(1 / (1 + distance))
                results.append(result)
        
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        prompt = build_prompt(query, context)
        
        try:
            import requests
            response = requests.post(
                f"{config.ollama_base_url}/api/generate",
                json={
                    "model": config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.llm_temperature,
                        "num_predict": config.llm_max_tokens
                    }
                },
                timeout=25
            )
            response.raise_for_status()
            answer = response.json()['response'].strip()
            logger.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:

        is_valid, error_msg = validate_query(question)
        if not is_valid:
            logger.warning(f"Invalid query: {error_msg}")
            return {
                'question': question,
                'answer': None,
                'retrieved_context': [],
                'error': error_msg
            }
        
        try:
            retrieved_chunks = self.retrieve(question, top_k)
            
            if not retrieved_chunks:
                logger.warning("No relevant context found")
                return {
                    'question': question,
                    'answer': "No relevant information found to answer this question.",
                    'retrieved_context': [],
                    'error': None
                }
            context = format_context(retrieved_chunks)
            answer = self.generate_answer(question, context)
            result = {
                'question': question,
                'answer': answer,
                'retrieved_context': [chunk['text'] for chunk in retrieved_chunks],
                'error': None
            }
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'question': question,
                'answer': None,
                'retrieved_context': [],
                'error': str(e)
            }