import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Union, List, Dict, Optional


class DebertaSmallClient:
    """DeBERTa Small NLI client for entailment scoring.
    
    Uses cross-encoder/nli-deberta-v3-small model for zero-shot classification.
    This model is significantly faster than BART-Large and specialized for NLI.
    """
    
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = None):
        """Initialize the DeBERTa Small client.
        
        Args:
            model_name: HuggingFace model name. Defaults to cross-encoder/nli-deberta-v3-small.
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
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Detect label indices dynamically
        self.entailment_idx = 1  # Default for this model but verifying
        self.contradiction_idx = 0
        
        if hasattr(self.model.config, "label2id"):
            l2id = self.model.config.label2id
            # Standard NLI labels: contradiction, entailment, neutral
            if "entailment" in l2id:
                self.entailment_idx = l2id["entailment"]
            if "contradiction" in l2id:
                self.contradiction_idx = l2id["contradiction"]
                
    def get_entailment_score(
        self, 
        node_info: str, 
        predicate: str,
        hypothesis_template: str = "This node {predicate}."
    ) -> float:
        """Calculate entailment score for a node given a predicate."""
        hypothesis = hypothesis_template.format(predicate=predicate)
        
        inputs = self.tokenizer(
            node_info, 
            hypothesis, 
            return_tensors="pt",
            truncation=True,
            max_length=128  # Optimization: reduced from 1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Select entailment vs contradiction only for normalized probability
        target_indices = [self.contradiction_idx, self.entailment_idx]
        entail_contradiction_logits = logits[:, target_indices]
        probs = torch.softmax(entail_contradiction_logits, dim=1)
        
        # Return probability of entailment (index 1 of our selected subset)
        return probs[0, 1].item()
    
    def batch_entailment_scores(
        self,
        node_infos: list[str],
        predicate: str,
        hypothesis_template: str = "This node {predicate}."
    ) -> list[float]:
        """Calculate entailment scores for multiple nodes against a single predicate."""
        hypothesis = hypothesis_template.format(predicate=predicate)
        
        # Optimization constants
        MAX_LENGTH = 128
        BATCH_SIZE = 16
        
        all_scores = []
        target_indices = [self.contradiction_idx, self.entailment_idx]
        
        # Process in chunks
        for i in range(0, len(node_infos), BATCH_SIZE):
            chunk_premises = node_infos[i : i + BATCH_SIZE]
            chunk_hypotheses = [hypothesis] * len(chunk_premises)
            
            inputs = self.tokenizer(
                chunk_premises,
                chunk_hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            entail_contradiction_logits = logits[:, target_indices]
            probs = torch.softmax(entail_contradiction_logits, dim=1)
            chunk_scores = probs[:, 1].tolist()
            all_scores.extend(chunk_scores)
            
            # Cleanup
            del inputs, outputs, logits
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return all_scores


# Singleton instance
_client_instance: DebertaSmallClient = None


def get_deberta_small_client(force_new: bool = False) -> DebertaSmallClient:
    """Get a DeBERTa Small client instance (singleton)."""
    global _client_instance
    if _client_instance is None or force_new:
        _client_instance = DebertaSmallClient()
    return _client_instance
