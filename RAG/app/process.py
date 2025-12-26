import pickle
from typing import List, Dict, Any
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from app import config
from app.utils import logger, clean_text, chunk_text

class DocumentIngestor:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
    def load_embedding_model(self):
        logger.info(f"Loading embedding model: {config.embedding_model_name}")
        self.embedding_model = SentenceTransformer(config.embedding_model_name)
        logger.info("Embedding model loaded successfully")
        
    def load_triviaqa_dataset(self) -> List[Dict[str, Any]]:
        logger.info(f"Loading TriviaQA dataset (max {config.max_documents} documents)")
        
        try:
            dataset = load_dataset(
                "trivia_qa",
                "unfiltered.nocontext",
                split=config.dataset_split
            )
            
            documents = []
            for idx, item in enumerate(dataset):
                if idx >= config.max_documents:
                    break
                
                question = item.get('question', '')
                answer_value = item.get('answer', {}).get('value', '')
                
                doc_text = clean_text(question)
                
                if len(doc_text) > 10:
                    documents.append({
                        'doc_id': f"doc_{idx}",
                        'text': doc_text,
                        'answer': answer_value,
                        'metadata': {
                            'question_id': item.get('question_id', ''),
                            'source': item.get('question_source', '')
                        }
                    })
            
            logger.info(f"Loaded {len(documents)} documents from TriviaQA")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> tuple[List[str], List[Dict]]:
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            text = doc['text']
            chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'doc_id': doc['doc_id'],
                    'chunk_id': f"{doc['doc_id']}_chunk_{chunk_idx}",
                    'text': chunk,
                    'answer': doc.get('answer', ''),
                    'metadata': doc.get('metadata', {})
                })
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks, all_metadata
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        logger.info("Embeddings generated successfully")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        logger.info("Building FAISS index")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self):
        config.index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = config.index_dir / "index.faiss"
        metadata_path = config.index_dir / "metadata.pkl"
        
        logger.info(f"Saving index to {index_path}")
        faiss.write_index(self.index, str(index_path))
        
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info("Index and metadata saved successfully")
    
    def ingest(self):
        try:
            self.load_embedding_model()
            raw_documents = self.load_triviaqa_dataset()
            chunks, metadata = self.process_documents(raw_documents)
            self.documents = chunks
            self.metadata = metadata
            embeddings = self.create_embeddings(chunks)
            self.build_faiss_index(embeddings)
            self.save_index()
            return True
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    ingestor.ingest()