"""
In-Context Evaluation - Direct LLM tree processing without semantic XPath.

This module provides infrastructure for evaluating LLM performance on
tree CRUD operations by passing the full tree directly to the model.
"""

from .in_context_pipeline import InContextPipeline
from .in_context_runner import InContextRunner
from .result_logger import InContextResultLogger

__all__ = [
    "InContextPipeline",
    "InContextRunner",
    "InContextResultLogger"
]
