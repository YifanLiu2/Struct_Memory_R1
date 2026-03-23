from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Any, Dict

from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import CRUDOperation


@dataclass
class ParsedQuery:
    """
    Parsed CRUD query result.

    Attributes:
        operation: The CRUD operation type
        xpath: The XPath query (without operation prefix)
        full_query: The complete query string with operation
        create_info: For CREATE operations, contains (parent_path, node_type, description)
        update_info: For UPDATE operations, contains (node_path, field_changes)
    """
    operation: CRUDOperation
    xpath: str
    full_query: str
    create_info: Optional[Tuple[str, str, str]] = None  # (parent_path, node_type, description)
    update_info: Optional[Tuple[str, Dict[str, Any]]] = None  # (node_path, changes)
    token_usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "operation": self.operation.value,
            "xpath": self.xpath,
            "full_query": self.full_query,
            "token_usage": self.token_usage
        }
        if self.create_info:
            result["create_info"] = {
                "parent_path": self.create_info[0],
                "node_type": self.create_info[1],
                "description": self.create_info[2]
            }
        if self.update_info:
            result["update_info"] = {
                "node_path": self.update_info[0],
                "changes": self.update_info[1]
            }
        return result