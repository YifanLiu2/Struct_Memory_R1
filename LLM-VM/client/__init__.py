"""
Client module - Unified access to LLM and embedding clients.

Main client:
- OpenAIClient: Unified client for GPT-4/GPT-5/o1/o3 models
  Configure model in config.yaml under openai.model

Embedding/NLI clients:
- BartNLIClient: BART-based NLI for entailment scoring
- TASBClient: TAS-B embeddings for cosine similarity
"""

from .openai_client import OpenAIClient, get_client, TokenUsage, CompletionResult
from .bart_client import BartNLIClient, get_bart_client
from .tas_b_client import TASBClient, get_tas_b_client, OpenAIEmbeddingClient
from .deberta_small_client import DebertaSmallClient, get_deberta_small_client
from .deberta_base_client import DebertaBaseClient, get_deberta_base_client


# Alias for backward compatibility
get_default_client = get_client


__all__ = [
    "OpenAIClient", "get_client", "get_default_client",
    "TokenUsage", "CompletionResult",
    "BartNLIClient", "get_bart_client",
    "TASBClient", "get_tas_b_client", "OpenAIEmbeddingClient",
    "DebertaSmallClient", "get_deberta_small_client",
    "DebertaBaseClient", "get_deberta_base_client",
]
