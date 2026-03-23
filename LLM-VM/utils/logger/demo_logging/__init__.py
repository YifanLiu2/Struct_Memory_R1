"""
Demo Logging - Specialized logger for visualization and debugging of semantic XPath execution.

Tracks:
- Child-to-parent score contributions for aggregation predicates
- Accumulated scores across traversal steps
- Predicate evaluation details for visualization
"""

from .demo_logger import DemoLogger, get_demo_logger, demo_log

__all__ = ["DemoLogger", "get_demo_logger", "demo_log"]
