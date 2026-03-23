"""
Tree Modification module for CRUD operations.

Provides:
- NodeDeleter: Delete nodes from XML trees
- NodeInserter: Insert new nodes into XML trees
- VersionManager: Manage versioned saves of trees
- ContentModifier: Manipulate version content (list of elements)
"""

from .base import (
    OperationType,
    OperationResult,
    TreeVersion,
    path_to_xpath,
    find_node_by_path,
    find_parent_and_index
)
from .node_deleter import NodeDeleter
from .node_inserter import NodeInserter
from .version_manager import VersionManager
from .content_modifier import (
    ContentModifier,
    copy_version_content,
    adjust_path_for_version,
    delete_from_content,
    replace_in_content,
    insert_in_content
)

__all__ = [
    # Base types
    "OperationType",
    "OperationResult",
    "TreeVersion",
    "path_to_xpath",
    "find_node_by_path",
    "find_parent_and_index",
    # Modifiers
    "NodeDeleter",
    "NodeInserter",
    "VersionManager",
    "ContentModifier",
    # Convenience functions
    "copy_version_content",
    "adjust_path_for_version",
    "delete_from_content",
    "replace_in_content",
    "insert_in_content"
]
