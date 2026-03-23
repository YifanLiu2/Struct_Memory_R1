"""
Node Deleter - Handles deletion of nodes from the XML tree.

Provides safe deletion with validation and rollback capability.
"""

import logging
import copy
from typing import List, Optional
import xml.etree.ElementTree as ET

from .base import (
    OperationResult, 
    OperationType, 
    find_node_by_path,
    find_parent_and_index
)


logger = logging.getLogger(__name__)


class NodeDeleter:
    """
    Handles deletion of nodes from XML trees.
    
    Supports:
    - Single node deletion
    - Multiple node deletion
    - Validation before deletion
    - Detailed operation tracing
    """
    
    def delete_node(
        self, 
        tree: ET.ElementTree, 
        tree_path: str
    ) -> OperationResult:
        """
        Delete a node from the tree by its path.
        
        Args:
            tree: The XML tree to modify
            tree_path: Path to the node to delete (e.g., "Itinerary > Day 1 > POI 2")
            
        Returns:
            OperationResult with success status and details
        """
        root = tree.getroot()
        
        # Find the node and its parent
        parent, child_index, node = find_parent_and_index(root, tree_path)
        
        if parent is None or node is None:
            return OperationResult(
                success=False,
                operation_type=OperationType.DELETE,
                node_path=tree_path,
                message=f"Node not found: {tree_path}",
                details={"error": "node_not_found"}
            )
        
        # Store node info before deletion for tracing
        node_info = self._node_to_dict(node)
        
        # Perform deletion
        try:
            parent.remove(node)
            
            return OperationResult(
                success=True,
                operation_type=OperationType.DELETE,
                node_path=tree_path,
                message=f"Successfully deleted node: {tree_path}",
                details={
                    "deleted_node": node_info,
                    "parent_tag": parent.tag,
                    "original_index": child_index
                },
                affected_nodes=[tree_path]
            )
            
        except Exception as e:
            logger.error(f"Error deleting node {tree_path}: {e}")
            return OperationResult(
                success=False,
                operation_type=OperationType.DELETE,
                node_path=tree_path,
                message=f"Error deleting node: {e}",
                details={"error": str(e)}
            )
    
    def delete_nodes(
        self, 
        tree: ET.ElementTree, 
        tree_paths: List[str]
    ) -> List[OperationResult]:
        """
        Delete multiple nodes from the tree.
        
        Note: Deletes are processed in reverse order of tree depth to avoid
        index shifting issues.
        
        Args:
            tree: The XML tree to modify
            tree_paths: List of paths to nodes to delete
            
        Returns:
            List of OperationResult for each deletion
        """
        # Sort paths by depth (deepest first) to avoid index issues
        sorted_paths = sorted(tree_paths, key=lambda p: p.count(">"), reverse=True)
        
        results = []
        for path in sorted_paths:
            result = self.delete_node(tree, path)
            results.append(result)
        
        return results
    
    def validate_deletion(
        self, 
        tree: ET.ElementTree, 
        tree_path: str
    ) -> tuple:
        """
        Validate that a node can be deleted.
        
        Args:
            tree: The XML tree
            tree_path: Path to the node to delete
            
        Returns:
            Tuple of (is_valid, reason)
        """
        root = tree.getroot()
        
        # Check if node exists
        node = find_node_by_path(root, tree_path)
        if node is None:
            return False, f"Node not found: {tree_path}"
        
        # Check if it's the root
        if node == root:
            return False, "Cannot delete root node"
        
        return True, "Deletion is valid"
    
    def _node_to_dict(self, node: ET.Element) -> dict:
        """Convert a node to a dictionary for tracing."""
        result = {
            "tag": node.tag,
            "attributes": dict(node.attrib)
        }
        
        # Add text content
        if node.text and node.text.strip():
            result["text"] = node.text.strip()
        
        # Add child elements' text as fields
        for child in node:
            if child.text and child.text.strip():
                result[child.tag] = child.text.strip()
            elif len(child) > 0:
                # Handle nested children (like highlights)
                result[child.tag] = [c.text for c in child if c.text]
        
        return result
