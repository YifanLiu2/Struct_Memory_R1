from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

class VersionSelector(Enum):
    """Version selector type."""
    AT = "at"        # Specific version: "in the version of xxx"
    BEFORE = "before"  # Version before: "rollback", "the version before"

class CRUDOperation(Enum):
    """CRUD operation types."""
    READ = "Read"
    CREATE = "Create"
    UPDATE = "Update"
    DELETE = "Delete"

@dataclass
class ResolvedVersion:
    """
    Result of version resolution.

    Attributes:
        selector_type: at or before
        semantic_query: Semantic description for version matching (None if using index)
        index: Numeric index (-1 for latest, or specific number)
        crud_operation: The CRUD operation type
        raw_response: Raw LLM response for debugging
        task_query: The task-relevant portion of the query (without version selection language)
    """
    selector_type: VersionSelector
    semantic_query: Optional[str]
    index: Optional[int]
    crud_operation: CRUDOperation
    raw_response: str
    task_query: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "selector_type": self.selector_type.value,
            "semantic_query": self.semantic_query,
            "index": self.index,
            "crud_operation": self.crud_operation.value,
            "raw_response": self.raw_response,
            "task_query": self.task_query,
            "token_usage": self.token_usage
        }

    def get_version_selector_string(self) -> str:
        """
        Get the version selector string for use in queries.

        Returns:
            Version selector string like "at([-1])" or "before(sem(content ~= 'museum'))"
        """
        if self.semantic_query:
            inner = f'sem(content ~= "{self.semantic_query}")'
        else:
            inner = f"[{self.index}]"

        return f"{self.selector_type.value}({inner})"