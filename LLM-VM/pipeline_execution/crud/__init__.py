"""
CRUD Handlers for semantic XPath tree operations.

Provides downstream task handlers for:
- Read operations
- Delete operations
- Update operations
- Create operations

Also provides tree modification capabilities via TreeModificationMixin.
"""

from .read_handler import ReadHandler
from .delete_handler import DeleteHandler
from .update_handler import UpdateHandler
from .create_handler import CreateHandler
from .tree_modification_mixin import TreeModificationMixin, ModificationResult
from .base import (
    HandlerResult,
    ReadResult,
    DeleteResult,
    UpdateResult,
    CreateResult,
    SelectedNode,
    UpdateItem
)

__all__ = [
    # Handlers
    "ReadHandler",
    "DeleteHandler",
    "UpdateHandler",
    "CreateHandler",
    # Mixin
    "TreeModificationMixin",
    "ModificationResult",
    # Result types
    "HandlerResult",
    "ReadResult",
    "DeleteResult",
    "UpdateResult",
    "CreateResult",
    "SelectedNode",
    "UpdateItem"
]
