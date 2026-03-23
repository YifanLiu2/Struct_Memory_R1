"""
Semantic XPath Execution - Core query execution engine with semantic scoring.

Modules:
- execution_models: Data classes for execution results, traces, and traversal
- dense_xpath_executor: Main executor orchestrating all components
- predicate_handler: Semantic predicate scoring (isinstance dispatch)
- index_handler: Positional index operations
- predicate_scorer: Scoring backends (LLM, entailment, cosine)
"""

# Execution models
from .execution_models import (
    NodeItem,
    MatchedNode,
    TraversalStep,
    ExecutionResult,
    StepContribution,
    NodeFusionTrace,
    ScoreFusionTrace,
    FinalFilteringTrace,
)

# Components
from .index_handler import IndexHandler
from .predicate_handler import PredicateHandler

# Main executor
from .dense_xpath_executor import DenseXPathExecutor

# Re-export parsing types for backward compatibility
from pipeline_execution.semantic_xpath_parsing import (
    QueryParser,
    get_parser,
    QueryStep,
    IndexRange,
    PredicateNode,
    AtomPredicate,
    AggExistsPredicate,
    AggPrevPredicate,
    AndPredicate,
    OrPredicate,
    NotPredicate,
)

# Re-export util types for backward compatibility
from pipeline_execution.semantic_xpath_util import (
    NodeUtils,
    load_schema,
    load_version_schema,
    get_data_path,
    get_schema_info,
    get_versioning_info,
    get_schema_summary_for_prompt,
    list_available_schemas,
    list_available_data_files,
)

# Re-export trace writer for backward compatibility
from utils.logger.query_execution_logging import TraceWriter

__all__ = [
    # Execution models
    "NodeItem",
    "MatchedNode",
    "TraversalStep",
    "ExecutionResult",
    "StepContribution",
    "NodeFusionTrace",
    "ScoreFusionTrace",
    "FinalFilteringTrace",
    # Components
    "IndexHandler",
    "PredicateHandler",
    # Main executor
    "DenseXPathExecutor",
    # Re-exported parsing types
    "QueryParser",
    "get_parser",
    "QueryStep",
    "IndexRange",
    "PredicateNode",
    "AtomPredicate",
    "AggExistsPredicate",
    "AggPrevPredicate",
    "AndPredicate",
    "OrPredicate",
    "NotPredicate",
    # Re-exported utils
    "NodeUtils",
    "TraceWriter",
    "load_schema",
    "load_version_schema",
    "get_data_path",
    "get_schema_info",
    "get_versioning_info",
    "get_schema_summary_for_prompt",
    "list_available_schemas",
    "list_available_data_files",
]
