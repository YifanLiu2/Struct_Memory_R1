"""
Version Manager - Manages in-tree versioning of XML trees.

Supports the new 2-stage query processing with 'at' and 'before' selectors.

Structure (schema-defined format):
Root
└── VersionNode (number="1")
    ├── patch_info: description of changes
    ├── conversation_history: user's request
    └── ContentContainer (content nodes)
└── VersionNode (number="2")
    ├── patch_info: "deleted item"
    ├── conversation_history: "delete the item"
    └── ContentContainer (modified content)

Also supports legacy format with a 'Version' tag for backward compatibility.
"""

import copy
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import xml.etree.ElementTree as ET

from .base import TreeVersion
from pipeline_execution.semantic_xpath_util import get_versioning_info


logger = logging.getLogger(__name__)


class VersionManager:
    """
    Manages in-tree versioning of XML trees.
    
    Versions are stored as child nodes of the root element.
    Each Itinerary_Version node contains:
    - number attribute: version number (1, 2, 3, ...)
    - patch_info: description of changes made
    - conversation_history: user's natural language request
    - Itinerary: container for content nodes (Days)
    """
    
    # Version tag names to support legacy formats
    VERSION_TAGS = ("Version",)
    
    def __init__(self, base_directory: Path = None, schema_name: Optional[str] = None):
        """
        Initialize the version manager.
        
        Args:
            base_directory: Directory for saving tree files.
                          If None, uses the original file's directory.
        """
        self.base_directory = base_directory
        self._versioning = get_versioning_info(schema_name)
        self._version_tag = self._versioning.get("version_tag")
        self._version_index_attr = self._versioning.get("version_index_attr", "number")
        self._content_container = self._versioning.get("content_container")
        self._metadata_tags = {"patch_info", "conversation_history"}
    
    def _find_versions(self, tree: ET.ElementTree) -> List[ET.Element]:
        """
        Find all version elements in the tree.
        
        Supports schema-defined version tags or legacy tags.
        
        Args:
            tree: The XML tree
            
        Returns:
            List of version elements
        """
        root = tree.getroot()
        versions = []
        
        # Prefer schema-defined version tag if available
        if self._version_tag:
            versions.extend(root.findall(self._version_tag))
            return versions
        
        for child in list(root):
            if self._is_version_tag(child.tag):
                versions.append(child)
        
        return versions
    
    def _get_version_tag(self, tree: ET.ElementTree) -> str:
        """
        Determine which version tag is used in this tree.
        
        Args:
            tree: The XML tree
            
        Returns:
            Version tag name (schema-defined or legacy "Version")
        """
        root = tree.getroot()
        
        if self._version_tag:
            return self._version_tag
        
        # Prefer a detected version tag from the tree
        for child in list(root):
            if self._is_version_tag(child.tag):
                return child.tag
        
        # Fall back to legacy format
        return "Version"

    @staticmethod
    def _is_version_tag(tag: str) -> bool:
        """Heuristic for legacy version tag detection when schema is missing."""
        if tag in VersionManager.VERSION_TAGS:
            return True
        return tag.endswith("_Version")
    
    def get_version_count(self, tree: ET.ElementTree) -> int:
        """
        Get the number of versions in the tree.
        
        Args:
            tree: The XML tree
            
        Returns:
            Number of Version nodes
        """
        return len(self._find_versions(tree))
    
    def get_latest_version(self, tree: ET.ElementTree) -> Optional[ET.Element]:
        """
        Get the latest (last) Version element from the tree.
        
        Args:
            tree: The XML tree
            
        Returns:
            The last Version element, or None if no versions exist
        """
        versions = self._find_versions(tree)
        
        if not versions:
            return None
        
        # Return the last version (highest number)
        return versions[-1]
    
    def get_version_by_number(self, tree: ET.ElementTree, number: int) -> Optional[ET.Element]:
        """
        Get a Version element by its number.
        
        Args:
            tree: The XML tree
            number: Version number (1-indexed, or negative for from-end)
            
        Returns:
            The Version element, or None if not found
        """
        versions = self._find_versions(tree)
        
        if not versions:
            return None
        
        # Handle negative indexing
        if number < 0:
            idx = len(versions) + number
            if 0 <= idx < len(versions):
                return versions[idx]
            return None
        
        # Positive number - find by @number attribute
        for version in versions:
            if version.get(self._version_index_attr) == str(number):
                return version
        
        return None
    
    def get_previous_version(
        self, 
        tree: ET.ElementTree, 
        current_version: ET.Element
    ) -> Optional[ET.Element]:
        """
        Get the version immediately before the given version.
        
        Used for 'before' selector in 2-stage query processing.
        
        Args:
            tree: The XML tree
            current_version: The reference version element
            
        Returns:
            The previous Version element, or None if current is first
        """
        versions = self._find_versions(tree)
        
        if not versions or len(versions) < 2:
            return None
        
        # Find the index of current version
        try:
            current_idx = versions.index(current_version)
        except ValueError:
            return None
        
        # Return previous version if it exists
        if current_idx > 0:
            return versions[current_idx - 1]
        
        return None
    
    def get_version_by_semantic(
        self,
        tree: ET.ElementTree,
        semantic_query: str,
        scorer=None
    ) -> Optional[ET.Element]:
        """
        Find a version by semantic matching on patch_info/conversation_history.
        
        Uses entailment scoring to find the best matching version (top-1).
        Falls back to keyword matching if no scorer is provided.
        
        Args:
            tree: The XML tree
            semantic_query: Natural language description to match
            scorer: Optional predicate scorer (EntailmentPredicateScorer)
            
        Returns:
            Best matching Version element, or None
        """
        versions = self._find_versions(tree)
        
        if not versions:
            return None
        
        # Build version nodes for scoring
        version_nodes = []
        for i, version in enumerate(versions):
            patch_info = version.find("patch_info")
            conv_history = version.find("conversation_history")
            
            # Combine text for description
            text_parts = []
            if patch_info is not None and patch_info.text:
                text_parts.append(f"Changes: {patch_info.text}")
            if conv_history is not None and conv_history.text:
                text_parts.append(f"Request: {conv_history.text}")
            
            description = " | ".join(text_parts) if text_parts else "(no changes recorded)"
            version_number = version.get(self._version_index_attr, str(i + 1))
            
            version_nodes.append({
                "id": f"version_{version_number}",
                "type": "Version",
                "name": f"Version {version_number}",
                "description": description
            })
        
        # Use entailment scorer if available
        if scorer is not None:
            try:
                batch_result = scorer.score_batch(version_nodes, semantic_query)
                
                # Find top-1 scoring version
                if batch_result.results:
                    best_result = max(batch_result.results, key=lambda r: r.score)
                    best_idx = next(
                        (i for i, r in enumerate(batch_result.results) if r.node_id == best_result.node_id),
                        None
                    )
                    
                    if best_idx is not None and best_result.score > 0:
                        logger.info(f"Semantic version match: Version {versions[best_idx].get(self._version_index_attr)} "
                                  f"(score: {best_result.score:.3f})")
                        return versions[best_idx]
            except Exception as e:
                logger.warning(f"Semantic version scoring failed, falling back to keyword: {e}")
        
        # Fallback: keyword matching
        best_version = None
        best_score = 0.0
        search_term_lower = semantic_query.lower()
        
        for i, version in enumerate(versions):
            description = version_nodes[i]["description"].lower()
            
            if search_term_lower in description:
                # Count occurrences as a simple score
                score = description.count(search_term_lower)
                if score > best_score:
                    best_score = score
                    best_version = version
        
        return best_version if best_version is not None else self.get_latest_version(tree)
    
    def get_version_by_selector(
        self, 
        tree: ET.ElementTree, 
        selector: str,
        scorer=None
    ) -> Optional[ET.Element]:
        """
        Get a Version element by a selector string (legacy support).
        
        Selectors:
        - "[-1]" or "[-N]": Negative index (from end)
        - "[N]": Positive version number
        - "[atom(content =~ '...')]": Semantic match on patch_info/conversation_history
        
        Args:
            tree: The XML tree
            selector: Selector string (e.g., "[-1]", "[2]", "[atom(content =~ 'museum')]")
            scorer: Optional predicate scorer for semantic matching
            
        Returns:
            The matched Version element, or None
        """
        selector = selector.strip()
        
        # Remove brackets if present
        if selector.startswith("[") and selector.endswith("]"):
            selector = selector[1:-1]
        
        # Check for negative index
        if selector.startswith("-") and selector[1:].isdigit():
            return self.get_version_by_number(tree, int(selector))
        
        # Check for positive number
        if selector.isdigit():
            return self.get_version_by_number(tree, int(selector))
        
        # Check for semantic predicate
        if selector.startswith("atom("):
            # Extract the search term
            match = re.search(r'=~\s*["\']([^"\']+)["\']', selector)
            if match:
                return self.get_version_by_semantic(tree, match.group(1), scorer)
        
        # Default to last version
        return self.get_latest_version(tree)
    
    def _get_version_by_semantic(
        self,
        tree: ET.ElementTree,
        predicate: str,
        scorer=None
    ) -> Optional[ET.Element]:
        """
        Find a version by semantic matching (legacy method).
        
        Args:
            tree: The XML tree
            predicate: Semantic predicate string like "atom(content =~ 'museum')"
            scorer: Optional predicate scorer (EntailmentPredicateScorer)
            
        Returns:
            Best matching Version element, or None
        """
        # Extract the search term from the predicate
        match = re.search(r'=~\s*["\']([^"\']+)["\']', predicate)
        if not match:
            return self.get_latest_version(tree)
        
        return self.get_version_by_semantic(tree, match.group(1), scorer)
    
    def copy_version_content(self, version_elem: ET.Element) -> ET.Element:
        """
        Create a deep copy of a Version element's content.
        
        Args:
            version_elem: The Version element to copy
            
        Returns:
            A deep copy of the Version element
        """
        return copy.deepcopy(version_elem)
    
    def create_new_version(
        self,
        tree: ET.ElementTree,
        source_version: ET.Element,
        patch_info: str,
        conversation_history: str,
        modified_content: List[ET.Element] = None
    ) -> ET.Element:
        """
        Create a new Version node and append it to the tree.
        
        Args:
            tree: The XML tree to modify
            source_version: The source Version element (content will be copied)
            patch_info: Description of changes made
            conversation_history: User's original request
            modified_content: Optional pre-modified content nodes (Days).
                            If None, copies content from source_version.
            
        Returns:
            The newly created Version element
        """
        root = tree.getroot()
        
        # Determine version tag and new number
        version_tag = self._get_version_tag(tree)
        versions = self._find_versions(tree)
        new_number = len(versions) + 1
        
        # Create new Version element
        new_version = ET.Element(version_tag)
        new_version.set(self._version_index_attr, str(new_number))
        
        # Add patch_info
        patch_elem = ET.SubElement(new_version, "patch_info")
        patch_elem.text = patch_info
        
        # Add conversation_history
        conv_elem = ET.SubElement(new_version, "conversation_history")
        conv_elem.text = conversation_history
        
        # Handle content based on format
        content_container = self._get_content_container(source_version)
        if content_container is not None:
            # Content wrapped in a container
            container_tag = content_container.tag
            new_container = ET.SubElement(new_version, container_tag)
            
            if modified_content is not None:
                for elem in modified_content:
                    new_container.append(copy.deepcopy(elem))
            else:
                for child in content_container:
                    new_container.append(copy.deepcopy(child))
        else:
            # Content directly under version
            if modified_content is not None:
                for elem in modified_content:
                    new_version.append(copy.deepcopy(elem))
            else:
                # Copy content from source version (excluding metadata)
                for child in source_version:
                    if child.tag not in self._metadata_tags:
                        new_version.append(copy.deepcopy(child))
        
        # Append to root
        root.append(new_version)
        
        logger.info(f"Created new version {new_number}")
        
        return new_version
    
    def get_version_content(self, version_elem: ET.Element) -> List[ET.Element]:
        """
        Get the content nodes from a Version element (excluding metadata).
        
        For new format, returns children of the Itinerary element.
        For legacy format, returns direct children excluding metadata.
        
        Args:
            version_elem: The Version element
            
        Returns:
            List of content child elements (not patch_info or conversation_history)
        """
        content_container = self._get_content_container(version_elem)
        if content_container is not None:
            return list(content_container)
        
        # Content directly under version
        return [
            child for child in version_elem 
            if child.tag not in self._metadata_tags
        ]

    def _get_content_container(self, version_elem: ET.Element) -> Optional[ET.Element]:
        """
        Get the content container element if one exists.
        
        Uses schema-defined content_container when available, otherwise
        falls back to a structural heuristic to support legacy trees.
        """
        if self._content_container:
            container = version_elem.find(self._content_container)
            if container is not None:
                return container
        
        # Heuristic: a single non-metadata child that has children
        non_meta = [c for c in version_elem if c.tag not in self._metadata_tags]
        if len(non_meta) == 1 and len(non_meta[0]) > 0:
            return non_meta[0]
        
        return None
    
    def save_tree(
        self,
        tree: ET.ElementTree,
        original_path: Path,
        custom_name: str = None
    ) -> TreeVersion:
        """
        Save the tree to disk.
        
        Args:
            tree: The XML tree to save
            original_path: Path to the original file
            custom_name: Optional custom filename (without .xml extension)
            
        Returns:
            TreeVersion with save info
        """
        original_path = Path(original_path)
        
        # Determine output directory
        if self.base_directory:
            output_dir = self.base_directory
        else:
            output_dir = original_path.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if custom_name:
            output_path = output_dir / f"{custom_name}.xml"
        else:
            output_path = output_dir / original_path.name
        
        # Get version count
        version_count = self.get_version_count(tree)
        
        # Save the tree
        try:
            # Use indent for readable output (Python 3.9+)
            try:
                ET.indent(tree.getroot())
            except AttributeError:
                self._indent_element(tree.getroot())
            
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            
            timestamp = datetime.now().isoformat()
            
            version_info = TreeVersion(
                version=version_count,
                path=str(output_path),
                timestamp=timestamp,
                operation="SAVE",
                changes={"total_versions": version_count}
            )
            
            logger.info(f"Saved tree with {version_count} versions to {output_path}")
            print(f"[Tree saved] {output_path} ({version_count} versions)")
            
            return version_info
            
        except Exception as e:
            logger.error(f"Error saving tree: {e}")
            raise
    
    def get_version_history(self, tree: ET.ElementTree) -> List[Dict[str, Any]]:
        """
        Get version history from the tree.
        
        Args:
            tree: The XML tree
            
        Returns:
            List of version info dictionaries
        """
        versions = self._find_versions(tree)
        
        history = []
        for version in versions:
            patch_info = version.find("patch_info")
            conv_history = version.find("conversation_history")
            
            history.append({
            "number": int(version.get(self._version_index_attr, 0)),
                "patch_info": patch_info.text if patch_info is not None else "",
                "conversation_history": conv_history.text if conv_history is not None else "",
                "content_count": len(self.get_version_content(version))
            })
        
        return history
    
    def _indent_element(self, elem: ET.Element, level: int = 0):
        """
        Add indentation to XML element (fallback for Python < 3.9).
        """
        indent = "\n" + "  " * level
        
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_element(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent


# Legacy compatibility - keep old methods that may be called
class LegacyVersionManager(VersionManager):
    """
    Compatibility layer for code that expects the old file-based versioning.
    """
    
    VERSION_PATTERN = re.compile(r"_v(\d+)\.xml$")
    
    def save_version(
        self,
        tree: ET.ElementTree,
        original_path: Path,
        operation: str,
        changes: dict = None,
        custom_name: str = None
    ) -> TreeVersion:
        """
        Legacy method - saves tree with version tracking.
        
        For backward compatibility, this just saves the tree.
        New code should use save_tree() instead.
        """
        return self.save_tree(tree, original_path, custom_name)
