"""
Semantic XPath Pipeline - CRUD operations on tree data with natural language.

This module provides a complete pipeline for executing CRUD operations on tree data
using natural language queries with semantic XPath.

Main components:
- SemanticXPathPipeline: Core pipeline for processing queries
- SemanticXPathCLI: Interactive command-line interface
- ResultFormatter: Format operation results for display
- SessionStatistics: Track session metrics
"""

from .semantic_xpath_pipeline import SemanticXPathPipeline
from .semantic_xpath_cli import SemanticXPathCLI
from .semantic_xpath_data_model import ResultFormatter, SessionStatistics

__all__ = [
    "SemanticXPathPipeline",
    "SemanticXPathCLI",
    "ResultFormatter",
    "SessionStatistics"
]
