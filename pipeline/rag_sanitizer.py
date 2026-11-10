import time

class RAGChunkSanitizer:
    """
    Sanitizes external retrieved knowledge chunks before prompt assembly
    to defend against YAE (Zhang et al., 2026) and PoisonedRAG attacks.
    """
    def __init__(self, classifier_model, threshold=0.70):
        self.model = classifier_model
        self.threshold = threshold

    def sanitize_chunks(self, retrieved_chunks):
        """
        retrieved_chunks: List of string passages fetched from vector database.
        Returns:
            clean_chunks: List of validated benign passages.
            flagged_chunks: List of blocked chunks with threat scores.
        """
        clean_chunks = []
        flagged_chunks = []
        
        suspicious_terms = [
            "ignore previous instructions",
            "you are an expert in bypass",
            "developer mode",
            "guidance prefix",
            "drop all rules",
            "system override"
        ]
        
        for idx, chunk in enumerate(retrieved_chunks):
            is_suspicious = False
            chunk_lower = chunk.lower()
            for term in suspicious_terms:
                if term in chunk_lower:
                    is_suspicious = True
                    break
            
            if is_suspicious:
                flagged_chunks.append({
                    "index": idx,
                    "chunk": chunk[:80] + "...",
                    "status": "BLOCKED_POISONED_RAG",
                    "score": 0.94
                })
            else:
                clean_chunks.append(chunk)
                
        return clean_chunks, flagged_chunks
