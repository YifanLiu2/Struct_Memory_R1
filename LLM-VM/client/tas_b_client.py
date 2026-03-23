import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Union, List

from .openai_client import OpenAIClient, load_config


class TASBClient:
    """TAS-B embedding client for dense retrieval.
    
    Uses sentence-transformers/msmarco-distilbert-base-tas-b (or compatible HF model)
    for generating text embeddings locally.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/msmarco-distilbert-base-tas-b",
        device: str = None
    ):
        """Initialize the TAS-B client.
        
        Args:
            model_name: HuggingFace model name. Defaults to TAS-B.
            device: Device to run the model on. Auto-detected if None.
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embedding."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_embedding(
        self, 
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Get embedding for a single text string."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().numpy()
    
    def get_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """Get embeddings for multiple texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0).numpy()
    
    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.get_embedding(text1, normalize=True)
        emb2 = self.get_embedding(text2, normalize=True)
        return float(np.dot(emb1, emb2))


class OpenAIEmbeddingClient:
    """Embedding client backed by OpenAI-compatible / OpenRouter embeddings.
    
    Uses the same OpenAIClient configuration (including base_url) and calls the
    embeddings endpoint for dense representations (e.g., Qwen3-8B embed).
    """
    
    def __init__(self, model_name: str, openai_config: dict):
        """
        Args:
            model_name: Embedding model ID to pass to embeddings.create().
            openai_config: A dict with OpenAI/OpenRouter settings for embeddings,
                           e.g. {"api_key": "...", "base_url": "..."}.
        """
        self.model_name = model_name
        # Wrap the provided openai_config in the structure OpenAIClient expects.
        self._client = OpenAIClient(config={"openai": openai_config})
        self.embedding_dim = None
    
    def get_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Get embedding for a single text string via embeddings.create()."""
        resp = self._client.client.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        vec = np.array(resp.data[0].embedding, dtype="float32")
        if self.embedding_dim is None:
            self.embedding_dim = vec.shape[0]
        if normalize:
            norm = np.linalg.norm(vec) + 1e-9
            vec = vec / norm
        return vec
    
    def get_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """Get embeddings for multiple texts via batched embeddings.create()."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            arr = np.array([d.embedding for d in resp.data], dtype="float32")
            if self.embedding_dim is None:
                self.embedding_dim = arr.shape[1]
            if normalize:
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
                arr = arr / norms
            all_vecs.append(arr)
        
        if not all_vecs:
            return np.zeros((0, self.embedding_dim or 0), dtype="float32")
        return np.vstack(all_vecs)
    
    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute cosine similarity between two texts using embeddings."""
        emb1 = self.get_embedding(text1, normalize=True)
        emb2 = self.get_embedding(text2, normalize=True)
        return float(np.dot(emb1, emb2))


# Singleton instance for convenience (legacy TAS-B usage)
_client_instance: TASBClient = None


def get_tas_b_client(force_new: bool = False) -> TASBClient:
    """Get a TAS-B client instance (singleton).
    
    This keeps the original behavior for callers that expect a local
    TAS-B model and are not wired into the experiment config system.
    """
    global _client_instance
    if _client_instance is None or force_new:
        _client_instance = TASBClient()
    return _client_instance


if __name__ == "__main__":
    # Quick test
    client = get_tas_b_client()
    
    # Test single embedding
    text = "A cozy restaurant with live jazz music"
    embedding = client.get_embedding(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10]}")
    
    # Test batch embeddings
    texts = [
        "Italian restaurant with pasta",
        "Jazz club with live music",
        "Quiet library for reading"
    ]
    embeddings = client.get_embeddings(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = client.similarity(
        "A restaurant with jazz music",
        "A place with live musical performances"
    )
    print(f"\nSimilarity score: {sim:.4f}")

