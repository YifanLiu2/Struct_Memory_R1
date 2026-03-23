"""
Content Modifier - Handles manipulation of version content (list of XML elements).

This module provides operations on version content without requiring the full tree,
enabling cleaner separation between orchestration and tree modification logic.
"""

import copy
import logging
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

from pipeline_execution.semantic_xpath_util import get_versioning_info


logger = logging.getLogger(__name__)


class ContentModifier:
    """
    Handles modification of version content (list of XML elements).
    
    Version content is the list of elements under a version's Itinerary container,
    typically Day elements with their children (POIs, Restaurants, etc.).
    
    This class provides methods to:
    - Find nodes within content by relative path
    - Delete nodes from content
    - Replace nodes in content
    - Insert nodes into content
    """
    
    @staticmethod
    def copy_version_content(
        version: ET.Element,
        schema_name: Optional[str] = None,
        config: Optional[dict] = None
    ) -> List[ET.Element]:
        """
        Create a deep copy of the content nodes from a version element.
        
        Args:
            version: The version element (Itinerary_Version or Version)
            schema_name: Schema name for versioning info resolution
            config: Configuration dict
            
        Returns:
            List of deep-copied content elements (e.g., Day elements)
        """
        metadata_tags = {"patch_info", "conversation_history"}
        
        # Prefer schema-defined content container when available
        versioning = get_versioning_info(schema_name, config)
        content_container = versioning.get("content_container")
        if content_container:
            container = version.find(content_container)
            if container is not None:
                return [copy.deepcopy(child) for child in container]
        
        # Heuristic: single non-metadata child with its own children
        non_meta = [c for c in version if c.tag not in metadata_tags]
        if len(non_meta) == 1 and len(non_meta[0]) > 0:
            return [copy.deepcopy(child) for child in non_meta[0]]
        
        # Content directly under version
        return [
            copy.deepcopy(child)
            for child in version
            if child.tag not in metadata_tags
        ]
    
    @staticmethod
    def adjust_path_for_version(
        tree_path: str,
        version: ET.Element = None,
        schema_name: Optional[str] = None,
        config: Optional[dict] = None
    ) -> str:
        """
        Adjust a full tree path to be relative to version content.
        
        Strips Root, Itinerary_Version, and Itinerary prefixes from paths.
        
        Examples:
            "Root > Itinerary_Version 1 > Itinerary > Day 1 > POI 2" -> "Day 1 > POI 2"
            "Itinerary > Day 1 > Royal Ontario Museum" -> "Day 1 > Royal Ontario Museum"
        
        Args:
            tree_path: Full tree path string
            version: Optional version element (unused, kept for API compatibility)
            schema_name: Schema name for versioning info resolution
            config: Configuration dict
            
        Returns:
            Relative path within version content
        """
        parts = [p.strip() for p in tree_path.split(">")]
        versioning = get_versioning_info(schema_name, config)
        version_tag = version.tag if version is not None else versioning.get("version_tag")
        path_prefix = versioning.get("version_path_parts") or []
        content_container = versioning.get("content_container")
        
        def strip_prefix(parts_list: List[str], prefix_parts: List[str]) -> List[str]:
            idx = 0
            for prefix in prefix_parts:
                if idx >= len(parts_list):
                    return parts_list
                if not parts_list[idx].startswith(prefix):
                    return parts_list
                idx += 1
            return parts_list[idx:]
        
        # If the version node appears in the path, strip up to it
        if version_tag:
            for i, part in enumerate(parts):
                if part.startswith(version_tag):
                    parts = parts[i + 1:]
                    break
        
        # Otherwise, try stripping the schema-defined root-to-version path
        if path_prefix and parts:
            parts = strip_prefix(parts, path_prefix)
        
        # Remove content container if defined and at start
        if content_container and parts and parts[0].startswith(content_container):
            parts = parts[1:]
        
        return " > ".join(parts)
    
    @classmethod
    def find_node_in_content(
        cls,
        content: List[ET.Element],
        relative_path: str
    ) -> Optional[ET.Element]:
        """
        Find a node within the content list by relative path.
        
        Args:
            content: List of content elements (e.g., Day elements)
            relative_path: Relative path within content (e.g., "Day 1 > POI 2")
            
        Returns:
            The found element, or None if not found
        """
        if not relative_path:
            return None
        
        parts = [p.strip() for p in relative_path.split(">")]
        
        # Find first element
        first_part = parts[0]
        current = cls._find_element_by_part(content, first_part)
        
        if current is None:
            return None
        
        # Navigate through remaining parts
        for part in parts[1:]:
            children = list(current)
            current = cls._find_element_by_part(children, part)
            if current is None:
                return None
        
        return current
    
    @classmethod
    def find_parent_in_content(
        cls,
        content: List[ET.Element],
        relative_path: str
    ) -> Tuple[Optional[List[ET.Element]], Optional[ET.Element], int]:
        """
        Find a node's parent within the content.
        
        Args:
            content: List of content elements
            relative_path: Relative path to the target node
            
        Returns:
            Tuple of (parent_container, node, index_in_parent):
            - For top-level nodes: (content list, node, index in content)
            - For nested nodes: (parent.children as list, node, index)
            - If not found: (None, None, -1)
        """
        if not relative_path:
            return None, None, -1
        
        parts = [p.strip() for p in relative_path.split(">")]
        
        if len(parts) == 1:
            # Top-level element
            node = cls._find_element_by_part(content, parts[0])
            if node is not None:
                idx = content.index(node)
                return content, node, idx
            return None, None, -1
        
        # Find parent
        parent_path = " > ".join(parts[:-1])
        parent = cls.find_node_in_content(content, parent_path)
        
        if parent is None:
            return None, None, -1
        
        # Find child
        child_part = parts[-1]
        child = cls._find_element_by_part(list(parent), child_part)
        
        if child is not None:
            idx = list(parent).index(child)
            return list(parent), child, idx
        
        return None, None, -1
    
    @classmethod
    def delete_from_content(
        cls,
        content: List[ET.Element],
        relative_path: str
    ) -> bool:
        """
        Delete a node from the content list.
        
        Args:
            content: List of content elements (modified in place)
            relative_path: Relative path to the node to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        if not relative_path:
            return False
        
        parts = [p.strip() for p in relative_path.split(">")]
        
        if len(parts) == 1:
            # Delete from top level
            old_elem = cls._find_element_by_part(content, parts[0])
            if old_elem is not None:
                content.remove(old_elem)
                logger.debug(f"Deleted top-level element: {parts[0]}")
                return True
            return False
        
        # Delete from nested location
        parent_path = " > ".join(parts[:-1])
        parent = cls.find_node_in_content(content, parent_path)
        
        if parent is None:
            logger.warning(f"Parent not found for deletion: {parent_path}")
            return False
        
        child_part = parts[-1]
        old_child = cls._find_element_by_part(list(parent), child_part)
        
        if old_child is not None:
            parent.remove(old_child)
            logger.debug(f"Deleted nested element: {relative_path}")
            return True
        
        logger.warning(f"Child not found for deletion: {child_part}")
        return False
    
    @classmethod
    def replace_in_content(
        cls,
        content: List[ET.Element],
        relative_path: str,
        new_element: ET.Element
    ) -> bool:
        """
        Replace a node in the content list.
        
        Args:
            content: List of content elements (modified in place)
            relative_path: Relative path to the node to replace
            new_element: The replacement element
            
        Returns:
            True if replacement succeeded, False otherwise
        """
        if not relative_path:
            return False
        
        parts = [p.strip() for p in relative_path.split(">")]
        
        if len(parts) == 1:
            # Replace at top level
            old_elem = cls._find_element_by_part(content, parts[0])
            if old_elem is not None:
                idx = content.index(old_elem)
                content[idx] = new_element
                logger.debug(f"Replaced top-level element: {parts[0]}")
                return True
            return False
        
        # Replace at nested location
        parent_path = " > ".join(parts[:-1])
        parent = cls.find_node_in_content(content, parent_path)
        
        if parent is None:
            logger.warning(f"Parent not found for replacement: {parent_path}")
            return False
        
        child_part = parts[-1]
        old_child = cls._find_element_by_part(list(parent), child_part)
        
        if old_child is not None:
            idx = list(parent).index(old_child)
            parent.remove(old_child)
            parent.insert(idx, new_element)
            logger.debug(f"Replaced nested element: {relative_path}")
            return True
        
        logger.warning(f"Child not found for replacement: {child_part}")
        return False
    
    @classmethod
    def insert_in_content(
        cls,
        content: List[ET.Element],
        relative_parent_path: str,
        new_element: ET.Element,
        position: int = -1
    ) -> bool:
        """
        Insert a new node into the content.
        
        Args:
            content: List of content elements (modified in place)
            relative_parent_path: Relative path to the parent node.
                                  Empty string means insert at top level.
            new_element: The new element to insert
            position: Position among siblings (-1 for append, 0 for beginning)
            
        Returns:
            True if insertion succeeded, False otherwise
        """
        if not relative_parent_path:
            # Insert at top level
            if position < 0:
                content.append(new_element)
            else:
                content.insert(position, new_element)
            logger.debug(f"Inserted element at top level, position {position}")
            return True
        
        # Find parent
        parent = cls.find_node_in_content(content, relative_parent_path)
        
        if parent is None:
            logger.warning(f"Parent not found for insertion: {relative_parent_path}")
            return False
        
        # Insert under parent
        if position < 0:
            parent.append(new_element)
        else:
            parent.insert(position, new_element)
        
        logger.debug(f"Inserted element under {relative_parent_path}, position {position}")
        return True
    
    @staticmethod
    def _find_element_by_part(
        elements: List[ET.Element],
        part: str
    ) -> Optional[ET.Element]:
        """
        Find an element in a list by a path part.
        
        Supports multiple path part formats:
        - Indexed notation: "Day 1", "POI 2"
        - Direct tag match: "Itinerary", "Day"
        - Name-based match: "Royal Ontario Museum", "St. Lawrence Market"
        
        Args:
            elements: List of elements to search
            part: Path part string
            
        Returns:
            The found element, or None
        """
        words = part.split()
        
        # Case 1: Indexed notation like "Day 1" or "POI 2"
        if len(words) == 2 and words[1].isdigit():
            node_type = words[0]
            index = int(words[1])
            
            # First try: find by @index or @number attribute
            for elem in elements:
                if elem.tag == node_type:
                    elem_index = elem.get("index") or elem.get("number")
                    if elem_index == str(index):
                        return elem
            
            # Second try: positional indexing among same-type siblings
            matching = [e for e in elements if e.tag == node_type]
            if 0 < index <= len(matching):
                return matching[index - 1]
            
            return None
        
        # Case 2: Direct tag match
        for elem in elements:
            if elem.tag == part:
                return elem
        
        # Case 3a: Name-based match via XML attribute (e.g., <Category name="Work">)
        for elem in elements:
            attr_name = elem.get("name")
            if attr_name and attr_name.strip() == part:
                return elem
        
        # Case 3b: Name-based match (check <name> and <title> sub-elements)
        for elem in elements:
            name_elem = elem.find("name")
            if name_elem is not None and name_elem.text and name_elem.text.strip() == part:
                return elem
            
            title_elem = elem.find("title")
            if title_elem is not None and title_elem.text and title_elem.text.strip() == part:
                return elem
        
        return None


# Convenience functions for common operations
def copy_version_content(
    version: ET.Element,
    schema_name: Optional[str] = None,
    config: Optional[dict] = None
) -> List[ET.Element]:
    """Convenience function for ContentModifier.copy_version_content."""
    return ContentModifier.copy_version_content(version, schema_name, config)


def adjust_path_for_version(
    tree_path: str,
    version: ET.Element = None,
    schema_name: Optional[str] = None,
    config: Optional[dict] = None
) -> str:
    """Convenience function for ContentModifier.adjust_path_for_version."""
    return ContentModifier.adjust_path_for_version(tree_path, version, schema_name, config)


def delete_from_content(content: List[ET.Element], relative_path: str) -> bool:
    """Convenience function for ContentModifier.delete_from_content."""
    return ContentModifier.delete_from_content(content, relative_path)


def replace_in_content(
    content: List[ET.Element],
    relative_path: str,
    new_element: ET.Element
) -> bool:
    """Convenience function for ContentModifier.replace_in_content."""
    return ContentModifier.replace_in_content(content, relative_path, new_element)


def insert_in_content(
    content: List[ET.Element],
    relative_parent_path: str,
    new_element: ET.Element,
    position: int = -1
) -> bool:
    """Convenience function for ContentModifier.insert_in_content."""
    return ContentModifier.insert_in_content(content, relative_parent_path, new_element, position)
