"""
Node Inserter - Handles insertion of new nodes into the XML tree.

Provides safe insertion with validation and position control.
"""

import logging
from typing import Optional
import xml.etree.ElementTree as ET

from .base import (
    OperationResult,
    OperationType,
    find_node_by_path
)


logger = logging.getLogger(__name__)


class NodeInserter:
    """
    Handles insertion of nodes into XML trees.
    
    Supports:
    - Insertion at specific positions
    - Append to end
    - Insert at beginning
    - Validation before insertion
    """
    
    def insert_node(
        self,
        tree: ET.ElementTree,
        parent_path: str,
        new_node: ET.Element,
        position: int = -1
    ) -> OperationResult:
        """
        Insert a new node into the tree.
        
        Args:
            tree: The XML tree to modify
            parent_path: Path to the parent node
            new_node: The new XML element to insert
            position: Position among siblings (-1 for append, 0 for beginning)
            
        Returns:
            OperationResult with success status and details
        """
        root = tree.getroot()
        
        # Find parent node
        if parent_path == "root" or parent_path == root.tag:
            parent = root
        else:
            parent = find_node_by_path(root, parent_path)
        
        if parent is None:
            return OperationResult(
                success=False,
                operation_type=OperationType.INSERT,
                node_path=parent_path,
                message=f"Parent node not found: {parent_path}",
                details={"error": "parent_not_found"}
            )
        
        try:
            # Determine insertion position
            num_children = len(parent)
            
            if position == -1:
                # Append at end
                parent.append(new_node)
                actual_position = num_children
            elif position >= num_children:
                # Position beyond end, append
                parent.append(new_node)
                actual_position = num_children
            else:
                # Insert at specific position
                parent.insert(position, new_node)
                actual_position = position
            
            # Calculate the new node's path
            new_node_type = new_node.tag
            same_type_count = len([c for c in parent if c.tag == new_node_type])
            new_node_path = f"{parent_path} > {new_node_type} {same_type_count}"
            
            node_info = self._node_to_dict(new_node)
            
            return OperationResult(
                success=True,
                operation_type=OperationType.INSERT,
                node_path=new_node_path,
                message=f"Successfully inserted {new_node_type} at position {actual_position}",
                details={
                    "parent_path": parent_path,
                    "position": actual_position,
                    "new_node": node_info
                },
                affected_nodes=[new_node_path]
            )
            
        except Exception as e:
            logger.error(f"Error inserting node at {parent_path}: {e}")
            return OperationResult(
                success=False,
                operation_type=OperationType.INSERT,
                node_path=parent_path,
                message=f"Error inserting node: {e}",
                details={"error": str(e)}
            )
    
    def insert_xml_string(
        self,
        tree: ET.ElementTree,
        parent_path: str,
        xml_string: str,
        position: int = -1
    ) -> OperationResult:
        """
        Insert a node from an XML string.
        
        Args:
            tree: The XML tree to modify
            parent_path: Path to the parent node
            xml_string: XML string representation of the new node
            position: Position among siblings (-1 for append)
            
        Returns:
            OperationResult with success status and details
        """
        try:
            new_node = ET.fromstring(xml_string)
            return self.insert_node(tree, parent_path, new_node, position)
        except ET.ParseError as e:
            logger.error(f"Invalid XML string: {e}")
            return OperationResult(
                success=False,
                operation_type=OperationType.INSERT,
                node_path=parent_path,
                message=f"Invalid XML string: {e}",
                details={"error": "invalid_xml", "xml_string": xml_string[:200]}
            )
    
    def replace_node(
        self,
        tree: ET.ElementTree,
        node_path: str,
        new_node: ET.Element
    ) -> OperationResult:
        """
        Replace an existing node with a new one (for updates).
        
        Args:
            tree: The XML tree to modify
            node_path: Path to the node to replace
            new_node: The replacement node
            
        Returns:
            OperationResult with success status and details
        """
        root = tree.getroot()
        
        # Find the node and its parent
        parts = [p.strip() for p in node_path.split(">")]
        
        if len(parts) < 2:
            return OperationResult(
                success=False,
                operation_type=OperationType.UPDATE,
                node_path=node_path,
                message="Cannot replace root node",
                details={"error": "cannot_replace_root"}
            )
        
        parent_path = " > ".join(parts[:-1])
        parent = find_node_by_path(root, parent_path)
        
        if parent is None:
            return OperationResult(
                success=False,
                operation_type=OperationType.UPDATE,
                node_path=node_path,
                message=f"Parent not found: {parent_path}",
                details={"error": "parent_not_found"}
            )
        
        # Find the old node
        old_node = find_node_by_path(root, node_path)
        if old_node is None:
            return OperationResult(
                success=False,
                operation_type=OperationType.UPDATE,
                node_path=node_path,
                message=f"Node not found: {node_path}",
                details={"error": "node_not_found"}
            )
        
        try:
            # Find index of old node
            old_index = list(parent).index(old_node)
            
            # Store old node info
            old_info = self._node_to_dict(old_node)
            
            # Remove old and insert new
            parent.remove(old_node)
            parent.insert(old_index, new_node)
            
            new_info = self._node_to_dict(new_node)
            
            return OperationResult(
                success=True,
                operation_type=OperationType.UPDATE,
                node_path=node_path,
                message=f"Successfully replaced node at {node_path}",
                details={
                    "old_node": old_info,
                    "new_node": new_info,
                    "position": old_index
                },
                affected_nodes=[node_path]
            )
            
        except Exception as e:
            logger.error(f"Error replacing node {node_path}: {e}")
            return OperationResult(
                success=False,
                operation_type=OperationType.UPDATE,
                node_path=node_path,
                message=f"Error replacing node: {e}",
                details={"error": str(e)}
            )
    
    def validate_insertion(
        self,
        tree: ET.ElementTree,
        parent_path: str,
        node_type: str
    ) -> tuple:
        """
        Validate that a node can be inserted at the given location.
        
        Args:
            tree: The XML tree
            parent_path: Path to the intended parent
            node_type: Type of node to insert
            
        Returns:
            Tuple of (is_valid, reason)
        """
        root = tree.getroot()
        
        # Check if parent exists
        if parent_path == "root" or parent_path == root.tag:
            parent = root
        else:
            parent = find_node_by_path(root, parent_path)
        
        if parent is None:
            return False, f"Parent not found: {parent_path}"
        
        # Could add schema validation here
        return True, "Insertion is valid"
    
    def _node_to_dict(self, node: ET.Element) -> dict:
        """Convert a node to a dictionary for tracing."""
        result = {
            "tag": node.tag,
            "attributes": dict(node.attrib)
        }
        
        if node.text and node.text.strip():
            result["text"] = node.text.strip()
        
        for child in node:
            if child.text and child.text.strip():
                result[child.tag] = child.text.strip()
            elif len(child) > 0:
                result[child.tag] = [c.text for c in child if c.text]
        
        return result
