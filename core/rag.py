import tempfile
from pathlib import Path
import hashlib
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _SENTENCE_AVAILABLE = False
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False
import numpy as np
try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except Exception:
    PdfReader = None
    _PYPDF_AVAILABLE = False
try:
    from docx import Document
    _DOCX_AVAILABLE = True
except Exception:
    Document = None
    _DOCX_AVAILABLE = False
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    _OLLAMA_AVAILABLE = False
class RAGPipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', llm_model='mistral', chunk_size=500):
        if _SENTENCE_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception:
                self.embedding_model = None
        else:
            self.embedding_model = None
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.index = None
        self.chunks = []
    
    def load_document(self, file):
        # Save uploaded file to temp
        ext = Path(file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = Path(tmp.name)
        
        # Extract text
        try:
            if ext == '.pdf':
                if _PYPDF_AVAILABLE:
                    reader = PdfReader(tmp_path)
                    text = ' '.join(p.extract_text() or '' for p in reader.pages)
                else:
                    try:
                        text = tmp_path.read_bytes().decode('utf-8', errors='ignore')
                    except Exception:
                        text = ''
            elif ext == '.docx':
                if _DOCX_AVAILABLE:
                    doc = Document(tmp_path)
                    text = '\n'.join(p.text for p in doc.paragraphs)
                else:
                    try:
                        text = tmp_path.read_bytes().decode('utf-8', errors='ignore')
                    except Exception:
                        text = ''
            else:
                text = tmp_path.read_text(encoding='utf-8', errors='ignore')
        finally:
            # Cleanup temp file
            try:
                tmp_path.unlink()
            except Exception:
                pass
        
        # Chunk (overlapping)
        words = text.split()
        step = self.chunk_size // 2
        self.chunks = []
        for i in range(0, max(1, len(words) - self.chunk_size), step):
            chunk = ' '.join(words[i:i+self.chunk_size])
            if chunk:
                self.chunks.append(chunk)
        if not self.chunks and words:
            self.chunks = [' '.join(words[:self.chunk_size])]
    
    def build_index(self):
        if not self.chunks:
            return
        if not _SENTENCE_AVAILABLE:
            raise RuntimeError('sentence-transformers is not installed; cannot build embeddings')
        if self.embedding_model is None:
            raise RuntimeError('embedding model is not initialized; sentence-transformers may be unavailable')
        if not _FAISS_AVAILABLE:
            raise RuntimeError('faiss is not installed; cannot build vector index')
        embeddings = self.embedding_model.encode(self.chunks, normalize_embeddings=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
    
    def query(self, query, top_k=4):
        if self.index is None:
            return "No document indexed.", []
        q_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        retrieved = [self.chunks[i] for i in indices[0] if i != -1]
        context = "\n\n---\n\n".join(retrieved)
        prompt = f"Answer based ONLY on the context below. If the answer is not in the context, say 'Not found'.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        if not _OLLAMA_AVAILABLE:
            return 'Ollama not available in this environment', retrieved
        try:
            resp = ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
            if isinstance(resp, dict) and 'message' in resp:
                answer = resp['message'].get('content', str(resp))
            else:
                answer = str(resp)
        except Exception as e:
            answer = f"Ollama error: {e}"
        return answer, retrieved
