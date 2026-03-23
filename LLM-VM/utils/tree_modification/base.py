"""
Base classes for tree modification operations.

Provides common data structures and utilities for modifying XML trees.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import xml.etree.ElementTree as ET


class OperationType(Enum):
    """Types of tree modification operations."""
    INSERT = "INSERT"
    DELETE = "DELETE"
    UPDATE = "UPDATE"


@dataclass
class OperationResult:
    """
    Result of a tree modification operation.
    
    Attributes:
        success: Whether the operation succeeded
        operation_type: Type of operation performed
        node_path: Path to the affected node
        message: Human-readable result message
        details: Additional operation-specific details
        affected_nodes: List of paths to all affected nodes
    """
    success: bool
    operation_type: OperationType
    node_path: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    affected_nodes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "operation_type": self.operation_type.value,
            "node_path": self.node_path,
            "message": self.message,
            "details": self.details,
            "affected_nodes": self.affected_nodes
        }


@dataclass
class TreeVersion:
    """
    Information about a tree version.
    
    Attributes:
        version: Version number
        path: Path to the versioned file
        timestamp: When the version was created
        operation: Operation that created this version
        changes: Summary of changes
    """
    version: int
    path: str
    timestamp: str
    operation: str
    changes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "path": self.path,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "changes": self.changes
        }


def path_to_xpath(tree_path: str) -> str:
    """
    Convert a tree path string to an XPath-like query.
    
    Examples:
        "Itinerary > Day 1 > POI 2" -> "/Itinerary/Day[1]/POI[2]"
        "Itinerary > Day 3 > Restaurant 1" -> "/Itinerary/Day[3]/Restaurant[1]"
    
    Args:
        tree_path: Human-readable tree path
        
    Returns:
        XPath-like query string
    """
    parts = [p.strip() for p in tree_path.split(">")]
    xpath_parts = []
    
    for part in parts:
        # Check for indexed notation like "Day 1" or "POI 2"
        words = part.split()
        if len(words) == 2 and words[1].isdigit():
            node_type = words[0]
            index = words[1]
            # Use bracket notation for all node types including Day
            xpath_parts.append(f"{node_type}[{index}]")
        else:
            xpath_parts.append(part)
    
    return "/" + "/".join(xpath_parts)


def find_node_by_path(root: ET.Element, tree_path: str) -> Optional[ET.Element]:
    """
    Find a node in the tree by its path string.
    
    Supports multiple path formats:
    - Indexed: "Itinerary > Day 1 > POI 2"
    - Name-based: "Itinerary > Day 2 > Royal Ontario Museum"
    - Mixed: "Itinerary > Day 1 > Art Gallery of Ontario"
    
    Args:
        root: Root element of the tree
        tree_path: Human-readable tree path
        
    Returns:
        The found element, or None if not found
    """
    parts = [p.strip() for p in tree_path.split(">")]
    current = root
    
    # Skip root if it matches
    if parts and parts[0] == root.tag:
        parts = parts[1:]
    
    for part in parts:
        words = part.split()
        
        # Case 1: Indexed notation like "Day 1" or "POI 2"
        if len(words) == 2 and words[1].isdigit():
            node_type = words[0]
            index = int(words[1])
            
            # Prefer matching by index/number attribute when available
            found = None
            for child in current:
                if child.tag == node_type:
                    child_index = child.get("index") or child.get("number")
                    if child_index == str(index):
                        found = child
                        break
            
            if found is not None:
                current = found
                continue
            
            # Fallback: positional indexing among same-type siblings (1-based)
            children = [c for c in current if c.tag == node_type]
            if 0 < index <= len(children):
                current = children[index - 1]
            else:
                return None
        else:
            # Case 2: Try direct child by tag name first
            found = current.find(part)
            
            if found is None:
                # Case 3: Try to find by <name> element content
                # This handles paths like "Royal Ontario Museum"
                found = _find_child_by_name(current, part)
            
            if found is None:
                return None
            current = found
    
    return current


def _find_child_by_name(parent: ET.Element, name: str) -> Optional[ET.Element]:
    """
    Find a child element by its <name> sub-element content.
    
    Args:
        parent: Parent element to search in
        name: The name to search for
        
    Returns:
        The found child element, or None
    """
    for child in parent:
        # Check if child has a <name> element
        name_elem = child.find("name")
        if name_elem is not None and name_elem.text:
            if name_elem.text.strip() == name:
                return child
        
        # Also check <title> for other schemas
        title_elem = child.find("title")
        if title_elem is not None and title_elem.text:
            if title_elem.text.strip() == name:
                return child
    
    return None


def find_parent_and_index(root: ET.Element, tree_path: str) -> tuple:
    """
    Find the parent of a node and the node's index within the parent.
    
    Supports multiple path formats:
    - Indexed: "Itinerary > Day 1 > POI 2"
    - Name-based: "Itinerary > Day 2 > Royal Ontario Museum"
    
    Args:
        root: Root element of the tree
        tree_path: Human-readable tree path to the child node
        
    Returns:
        Tuple of (parent_element, child_index, child_element) or (None, -1, None)
    """
    parts = [p.strip() for p in tree_path.split(">")]
    
    if len(parts) < 2:
        return None, -1, None
    
    # Find parent
    parent_path = " > ".join(parts[:-1])
    parent = find_node_by_path(root, parent_path)
    
    if parent is None:
        return None, -1, None
    
    # Find child and its index
    child_part = parts[-1]
    words = child_part.split()
    
    # Case 1: Indexed notation like "POI 2"
    if len(words) == 2 and words[1].isdigit():
        node_type = words[0]
        target_index = int(words[1])
        
        # Find the child and its actual index
        for i, child in enumerate(parent):
            if child.tag == node_type:
                # Check if this is the right one
                children_of_type = [c for c in parent if c.tag == node_type]
                child_idx_in_type = children_of_type.index(child) + 1
                
                if child_idx_in_type == target_index:
                    return parent, i, child
    else:
        # Case 2: Name-based path like "Royal Ontario Museum"
        # First try direct tag name
        for i, child in enumerate(parent):
            if child.tag == child_part:
                return parent, i, child
        
        # Then try finding by <name> or <title> element
        for i, child in enumerate(parent):
            name_elem = child.find("name")
            if name_elem is not None and name_elem.text:
                if name_elem.text.strip() == child_part:
                    return parent, i, child
            
            title_elem = child.find("title")
            if title_elem is not None and title_elem.text:
                if title_elem.text.strip() == child_part:
                    return parent, i, child
    
    return None, -1, None
