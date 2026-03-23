"""
Tree Modification Mixin - Provides tree modification capabilities for CRUD handlers.

This mixin enables handlers to apply their modifications to version content,
keeping all CRUD-specific tree manipulation logic within the handlers.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET

from utils.tree_modification import (
    ContentModifier,
    copy_version_content,
    adjust_path_for_version
)


logger = logging.getLogger(__name__)


@dataclass
class ModificationResult:
    """
    Result of applying tree modifications.
    
    Attributes:
        success: Whether modifications were applied successfully
        modified_content: The modified content (list of elements)
        affected_paths: List of tree paths that were modified
        patch_info: Description of changes for version metadata
        error: Error message if failed
    """
    success: bool
    modified_content: Optional[List[ET.Element]] = None
    affected_paths: List[str] = None
    patch_info: str = ""
    error: str = ""
    
    def __post_init__(self):
        if self.affected_paths is None:
            self.affected_paths = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "affected_paths": self.affected_paths,
            "patch_info": self.patch_info,
            "error": self.error
        }


class TreeModificationMixin:
    """
    Mixin that provides tree modification capabilities for CRUD handlers.
    
    Provides methods for handlers to apply their decisions (delete/update/insert)
    to version content without needing to know the low-level tree manipulation details.
    """
    
    @staticmethod
    def prepare_version_content(
        version: ET.Element,
        schema_name: Optional[str] = None,
        config: Optional[dict] = None
    ) -> List[ET.Element]:
        """
        Prepare a copy of version content for modification.
        
        Args:
            version: The source version element
            schema_name: Schema name for versioning info resolution
            config: Configuration dict
            
        Returns:
            Deep copy of version content elements
        """
        return copy_version_content(version, schema_name, config)
    
    @staticmethod
    def adjust_path(
        tree_path: str,
        version: ET.Element = None,
        schema_name: Optional[str] = None,
        config: Optional[dict] = None
    ) -> str:
        """
        Adjust a tree path to be relative to version content.
        
        Args:
            tree_path: Full tree path
            version: Optional version element
            schema_name: Schema name for versioning info resolution
            config: Configuration dict
            
        Returns:
            Relative path within version content
        """
        return adjust_path_for_version(tree_path, version, schema_name, config)
    
    @staticmethod
    def apply_deletions(
        content: List[ET.Element],
        tree_paths: List[str],
        version: ET.Element = None,
        schema_name: Optional[str] = None,
        config: Optional[dict] = None
    ) -> ModificationResult:
        """
        Apply deletion operations to version content.
        
        Args:
            content: Version content elements (modified in place)
            tree_paths: List of tree paths to delete
            version: Optional version element for path adjustment
            schema_name: Schema name for versioning info resolution
            config: Configuration dict
            
        Returns:
            ModificationResult with success status and affected paths
        """
        if not tree_paths:
            return ModificationResult(
                success=True,
                modified_content=content,
                affected_paths=[],
                patch_info="No nodes to delete"
            )
        
        deleted_paths = []
        
        for tree_path in tree_paths:
            relative_path = adjust_path_for_version(tree_path, version, schema_name, config)
            
            if ContentModifier.delete_from_content(content, relative_path):
                deleted_paths.append(tree_path)
                logger.debug(f"Deleted: {tree_path}")
            else:
                logger.warning(f"Failed to delete: {tree_path}")
        
        if not deleted_paths:
            return ModificationResult(
                success=False,
                modified_content=content,
                error="No nodes were successfully deleted"
            )
        
        # Build patch info
        if len(deleted_paths) == 1:
            patch_info = f"Deleted: {deleted_paths[0]}"
        else:
            patch_info = f"Deleted {len(deleted_paths)} nodes: {', '.join(deleted_paths)}"
        
        return ModificationResult(
            success=True,
            modified_content=content,
            affected_paths=deleted_paths,
            patch_info=patch_info
        )
    
    @staticmethod
    def apply_updates(
        content: List[ET.Element],
        updates: List[Dict[str, Any]],
        version: ET.Element = None,
        schema_name: Optional[str] = None,
        config: Optional[dict] = None
    ) -> ModificationResult:
        """
        Apply update operations to version content.
        
        Args:
            content: Version content elements (modified in place)
            updates: List of update dicts with 'tree_path' and 'updated_content' keys
            version: Optional version element for path adjustment
            schema_name: Schema name for versioning info resolution
            config: Configuration dict
            
        Returns:
            ModificationResult with success status and affected paths
        """
        if not updates:
            return ModificationResult(
                success=True,
                modified_content=content,
                affected_paths=[],
                patch_info="No nodes to update"
            )
        
        updated_paths = []
        
        for update in updates:
            tree_path = update.get("tree_path", "")
            new_element = update.get("updated_content")
            
            if not tree_path or new_element is None:
                continue
            
            relative_path = adjust_path_for_version(tree_path, version, schema_name, config)
            
            if ContentModifier.replace_in_content(content, relative_path, new_element):
                updated_paths.append(tree_path)
                logger.debug(f"Updated: {tree_path}")
            else:
                logger.warning(f"Failed to update: {tree_path}")
        
        if not updated_paths:
            return ModificationResult(
                success=False,
                modified_content=content,
                error="No nodes were successfully updated"
            )
        
        # Build patch info
        if len(updated_paths) == 1:
            patch_info = f"Updated: {updated_paths[0]}"
        else:
            patch_info = f"Updated {len(updated_paths)} nodes: {', '.join(updated_paths)}"
        
        return ModificationResult(
            success=True,
            modified_content=content,
            affected_paths=updated_paths,
            patch_info=patch_info
        )
    
    @staticmethod
    def apply_insertion(
        content: List[ET.Element],
        parent_path: str,
        new_element: ET.Element,
        position: int = -1,
        version: ET.Element = None,
        schema_name: Optional[str] = None,
        config: Optional[dict] = None
    ) -> ModificationResult:
        """
        Apply an insertion operation to version content.
        
        Args:
            content: Version content elements (modified in place)
            parent_path: Path to the parent node
            new_element: The new element to insert
            position: Position among siblings (-1 for append)
            version: Optional version element for path adjustment
            schema_name: Schema name for versioning info resolution
            config: Configuration dict
            
        Returns:
            ModificationResult with success status and created path
        """
        if new_element is None:
            return ModificationResult(
                success=False,
                modified_content=content,
                error="No element to insert"
            )
        
        relative_parent_path = adjust_path_for_version(parent_path, version, schema_name, config) if parent_path else ""
        
        if ContentModifier.insert_in_content(content, relative_parent_path, new_element, position):
            # Construct the created path
            if relative_parent_path:
                created_path = f"{parent_path}/{new_element.tag}"
            else:
                created_path = new_element.tag
            
            patch_info = f"Created: {created_path}"
            
            return ModificationResult(
                success=True,
                modified_content=content,
                affected_paths=[created_path],
                patch_info=patch_info
            )
        
        return ModificationResult(
            success=False,
            modified_content=content,
            error=f"Failed to insert at {parent_path or 'root'}"
        )
