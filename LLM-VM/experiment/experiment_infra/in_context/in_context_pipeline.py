"""
In-Context Pipeline - Direct LLM tree processing for CRUD operations.

This pipeline sends the complete tree to the LLM along with user queries,
allowing the model to directly determine operations and modifications.

For multi-turn sessions, the tree accumulates versions and the LLM
autonomously decides which version to operate on based on the query.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from client import OpenAIClient, get_default_client
from utils.tree_modification import VersionManager
from pipeline_execution.semantic_xpath_util import load_schema, get_versioning_info, get_data_path


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class InContextResult:
    """Result from in-context pipeline processing."""
    success: bool
    operation: str
    version_used: int
    reasoning: str
    result_xml: str
    selected_nodes: List[str]
    related_nodes: List[str]
    token_usage: Dict[str, int]
    execution_time_ms: float
    raw_response: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "version_used": self.version_used,
            "reasoning": self.reasoning,
            "result_xml": self.result_xml,
            "selected_nodes": self.selected_nodes,
            "related_nodes": self.related_nodes,
            "token_usage": self.token_usage,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error
        }


class InContextPipeline:
    """
    Pipeline for direct LLM tree processing.
    
    Sends the complete tree XML to the LLM along with user queries.
    The LLM determines:
    - CRUD operation type
    - Which version to operate on (for multi-version trees)
    - The resulting XML (modified content or read results)
    
    Tree modifications are applied mechanically using VersionManager.
    """
    
    def __init__(self, tree_path: Path = None, config: Dict[str, Any] = None):
        """
        Initialize the in-context pipeline.
        
        Args:
            tree_path: Optional path to the XML tree file.
                      If None, will be set per-session.
            config: Optional experiment config dict. If provided, overrides
                   config.yaml for model, schema, and data settings.
        """
        self._config = config
        if config:
            self.client = OpenAIClient(config=config)
        else:
            self.client = get_default_client()
        self._schema = load_schema(config=config)
        self._versioning = get_versioning_info(config=config)
        self.version_manager = VersionManager(schema_name=self._schema.get("name"))
        self.prompt_template = self._load_prompt_template()
        
        self._tree_path = tree_path
        self._tree: Optional[ET.ElementTree] = None
        
        self._node_configs = self._schema.get("nodes", {})
        self._content_root = next(
            (name for name, cfg in self._node_configs.items() if cfg.get("type") == "root"),
            None
        )
        self._leaf_types = {name for name, cfg in self._node_configs.items() if cfg.get("type") == "leaf"}
        self._container_types = {
            name for name, cfg in self._node_configs.items()
            if cfg.get("type") in ("container", "root")
        }
        self._field_names = {
            field for cfg in self._node_configs.values() for field in cfg.get("fields", [])
        }
    
    def _load_prompt_template(self) -> str:
        """Load the in-context pipeline prompt template."""
        prompt_path = PROJECT_ROOT / "storage" / "prompts" / "experiment" / "incontext_pipeline.txt"
        with open(prompt_path, "r") as f:
            return f.read()
    
    def set_tree_path(self, tree_path: Path):
        """
        Set the tree path and load the tree.
        
        Args:
            tree_path: Path to the XML tree file
        """
        self._tree_path = Path(tree_path)
        self._tree = ET.parse(self._tree_path)
    
    @property
    def tree(self) -> ET.ElementTree:
        """Get the current tree."""
        if self._tree is None and self._tree_path:
            self._tree = ET.parse(self._tree_path)
        return self._tree
    
    def reload_tree(self):
        """Reload tree from file."""
        if self._tree_path:
            self._tree = ET.parse(self._tree_path)
    
    def process_request(
        self,
        user_query: str,
        tree_xml: str = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> InContextResult:
        """
        Process a user query against the tree.
        
        Args:
            user_query: Natural language query from user
            tree_xml: Optional tree XML string. If None, reads from tree_path.
            conversation_history: Optional list of previous turns for multi-turn.
                Each entry is {"role": "user"|"assistant", "content": "..."}.
                When provided, these are inserted between the system prompt
                and the current user message to give the LLM conversation context.
            
        Returns:
            InContextResult with operation details and results
        """
        start_time = time.perf_counter()
        
        # Get tree XML
        if tree_xml is None:
            if self._tree_path:
                with open(self._tree_path, "r") as f:
                    tree_xml = f.read()
            else:
                return InContextResult(
                    success=False,
                    operation="UNKNOWN",
                    version_used=0,
                    reasoning="No tree provided",
                    result_xml="",
                    selected_nodes=[],
                    related_nodes=[],
                    token_usage={},
                    execution_time_ms=0,
                    raw_response="",
                    error="No tree XML provided or tree_path set"
                )
        
        # Call LLM
        try:
            # Construct the user message with tree and query
            user_message = f"""## Current Tree State

{tree_xml}

## User Request

{user_query}"""
            
            # Build messages: system + optional conversation history + current query
            messages = [
                {"role": "system", "content": self.prompt_template},
            ]
            
            # Append multi-turn conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            messages.append({"role": "user", "content": user_message})
            
            completion_result = self.client.chat_with_usage(messages)
            
            raw_response = completion_result.content
            token_usage = completion_result.usage.to_dict()
            
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return InContextResult(
                success=False,
                operation="UNKNOWN",
                version_used=0,
                reasoning="",
                result_xml="",
                selected_nodes=[],
                related_nodes=[],
                token_usage={},
                execution_time_ms=execution_time_ms,
                raw_response="",
                error=f"LLM call failed: {str(e)}"
            )
        
        # Parse response
        parsed = self._parse_response(raw_response)
        
        # Extract related node paths from result
        related_nodes = self._extract_node_paths(
            parsed.get("operation", "UNKNOWN"),
            parsed.get("result", ""),
            parsed.get("reasoning", "")
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Use LLM-provided selected_nodes; fall back to extracted paths
        selected_nodes = parsed.get("selected_nodes", [])
        
        return InContextResult(
            success=parsed.get("success", True),
            operation=parsed.get("operation", "UNKNOWN"),
            version_used=parsed.get("version_used", 1),
            reasoning=parsed.get("reasoning", ""),
            result_xml=parsed.get("result", ""),
            selected_nodes=selected_nodes,
            related_nodes=related_nodes,
            token_usage=token_usage,
            execution_time_ms=execution_time_ms,
            raw_response=raw_response,
            error=parsed.get("error")
        )
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract JSON.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed dict with operation, version_used, reasoning, result
        """
        # Try to find JSON in the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'\{[^{}]*"operation"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try parsing entire response as JSON
                json_str = response.strip()
        
        try:
            parsed = json.loads(json_str)
            return {
                "success": True,
                "operation": parsed.get("operation", "UNKNOWN"),
                "version_used": parsed.get("version_used", 1),
                "reasoning": parsed.get("reasoning", ""),
                "selected_nodes": parsed.get("selected_nodes", []),
                "result": parsed.get("result", "")
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "operation": "UNKNOWN",
                "version_used": 1,
                "reasoning": "",
                "selected_nodes": [],
                "result": "",
                "error": f"Failed to parse JSON response: {str(e)}"
            }
    
    def _extract_node_paths(
        self, 
        operation: str, 
        result_xml: str, 
        reasoning: str
    ) -> List[str]:
        """
        Extract node paths from the LLM result.
        
        For READ: Parse result XML to find node paths
        For CUD: Parse reasoning and/or result to find affected paths
        
        Args:
            operation: CRUD operation type
            result_xml: XML result from LLM
            reasoning: LLM's reasoning text
            
        Returns:
            List of node path strings (e.g., "Root > Day 1 > POI")
        """
        paths = []
        
        if not result_xml:
            return paths
        
        try:
            # Wrap in root if needed for parsing
            if not result_xml.strip().startswith("<?xml"):
                parse_xml = f"<root>{result_xml}</root>"
            else:
                parse_xml = result_xml
            
            root = ET.fromstring(parse_xml)
            
            # Extract paths based on operation type
            if operation == "READ":
                # For READ, extract paths from returned nodes
                paths = self._extract_paths_from_elements(root)
            else:
                # For CUD, try to extract from reasoning
                paths = self._extract_paths_from_reasoning(reasoning)
                
                # If no paths in reasoning, try to find Day/POI structure
                if not paths:
                    paths = self._extract_paths_from_elements(root)
            
        except ET.ParseError:
            # If XML parsing fails, try to extract from reasoning
            paths = self._extract_paths_from_reasoning(reasoning)
        
        return paths
    
    def _extract_paths_from_elements(self, root: ET.Element) -> List[str]:
        """Extract node paths by traversing XML structure."""
        paths = []
        
        def build_path(elem: ET.Element, parent_path: str = ""):
            """Recursively build paths for elements."""
            tag = elem.tag
            if tag == "root":
                # Skip wrapper root
                for child in elem:
                    build_path(child, "")
                return
            
            # Build current path
            index = elem.get("index") or elem.get("number")
            if index:
                current_path = f"{parent_path} > {tag} {index}" if parent_path else f"{tag} {index}"
            else:
                name_elem = elem.find("name")
                if name_elem is not None and name_elem.text:
                    current_path = f"{parent_path} > {name_elem.text}" if parent_path else name_elem.text
                else:
                    current_path = f"{parent_path} > {tag}" if parent_path else tag
            
            # Add leaf nodes to paths
            if tag in self._leaf_types:
                paths.append(current_path.strip(" >"))
            elif tag in self._container_types:
                # Add Day to paths for READ queries about days
                paths.append(current_path.strip(" >"))
            
            # Recurse to children
            for child in elem:
                if child.tag not in self._field_names and child.tag not in ("patch_info", "conversation_history"):
                    build_path(child, current_path)
        
        build_path(root)
        return paths
    
    def _extract_paths_from_reasoning(self, reasoning: str) -> List[str]:
        """Extract node paths from reasoning text."""
        paths = []
        
        # Pattern to match paths like "Day 1 > POI Name" with dynamic version/root tags
        root_tag = self._versioning.get("root_tag")
        version_tag = self._versioning.get("version_tag")
        content_root = self._content_root or "Day"
        
        root_part = rf"(?:{root_tag}\s*>\s*)?" if root_tag else ""
        version_part = rf"(?:{version_tag}\s*\d+\s*>\s*)?" if version_tag else ""
        content_part = rf"{content_root}(?:\s*\d+)?(?:\s*>\s*[^,\n\]]+)?"
        
        path_pattern = rf"{root_part}{version_part}({content_part})"
        
        matches = re.findall(path_pattern, reasoning, re.IGNORECASE)
        for match in matches:
            path = match.strip()
            if path and path not in paths:
                paths.append(path)
        
        return paths
    
    def _unwrap_version_nodes(self, elements: List[ET.Element]) -> List[ET.Element]:
        """
        Recursively strip version wrapper nodes from LLM output.
        
        The LLM sometimes returns content wrapped in version tags like:
            <Itinerary_Version number="1"><Day>...</Day></Itinerary_Version>
        
        This method unwraps them to extract just the content nodes (Days),
        preventing nested versions like:
            Version 2 > Version 1 > Day ...  (wrong)
        Instead producing:
            Version 2 > Day ...              (correct)
        
        Args:
            elements: List of parsed XML elements from the LLM result
            
        Returns:
            Flat list of content nodes with all version wrappers removed
        """
        metadata_tags = self.version_manager._metadata_tags
        result = []
        for elem in elements:
            if VersionManager._is_version_tag(elem.tag):
                # This is a version wrapper — extract its non-metadata children
                children = [c for c in elem if c.tag not in metadata_tags]
                # Recursively unwrap in case of multiple layers of nesting
                result.extend(self._unwrap_version_nodes(children))
            else:
                result.append(elem)
        return result
    
    def apply_modifications(
        self,
        result: InContextResult,
        tree: ET.ElementTree,
        user_query: str
    ) -> Tuple[bool, ET.ElementTree]:
        """
        Apply CUD modifications to the tree.
        
        Creates a new version with the modified content.
        
        Args:
            result: InContextResult with the LLM response
            tree: Current XML tree
            user_query: Original user query (for conversation history)
            
        Returns:
            Tuple of (success, modified_tree)
        """
        if result.operation == "READ":
            # No modifications for READ
            return True, tree
        
        if not result.result_xml:
            return False, tree
        
        try:
            # Parse the result XML (should be complete Itinerary content)
            result_xml = result.result_xml.strip()
            
            # Strip XML declaration if present (e.g. <?xml version='1.0' encoding='utf-8'?>)
            # It can't appear inside a wrapper element and causes a parse error
            result_xml = re.sub(r'<\?xml[^?]*\?>\s*', '', result_xml)
            
            # Also strip any <Root>...</Root> wrapper the LLM may have included
            root_match = re.match(r'^\s*<Root\b[^>]*>(.*)</Root>\s*$', result_xml, re.DOTALL)
            if root_match:
                result_xml = root_match.group(1).strip()
            
            # Wrap in temporary root for parsing
            if not result_xml.startswith("<"):
                return False, tree
            
            # Parse the new content
            # The result should be the Itinerary children (Days)
            wrapped = f"<TempRoot>{result_xml}</TempRoot>"
            new_content_root = ET.fromstring(wrapped)
            new_content = list(new_content_root)
            
            if not new_content:
                return False, tree
            
            # Strip any version wrappers the LLM may have included.
            # The LLM is instructed NOT to include version wrappers, but
            # in multi-turn sessions it often returns content wrapped in
            # <Itinerary_Version> tags. Unwrapping prevents nested versions.
            new_content = self._unwrap_version_nodes(new_content)
            
            if not new_content:
                return False, tree
            
            # Get the source version (the one LLM operated on)
            source_version = self.version_manager.get_version_by_number(
                tree, result.version_used
            )
            
            if source_version is None:
                source_version = self.version_manager.get_latest_version(tree)
            
            if source_version is None:
                return False, tree
            
            # Create patch info based on operation
            patch_info = f"{result.operation}: {result.reasoning[:100]}" if result.reasoning else result.operation
            
            # Create new version
            self.version_manager.create_new_version(
                tree,
                source_version,
                patch_info=patch_info,
                conversation_history=user_query,
                modified_content=new_content
            )
            
            return True, tree
            
        except Exception as e:
            print(f"Error applying modifications: {e}")
            return False, tree
    
    def save_tree(self, tree: ET.ElementTree, output_path: Path):
        """
        Save the tree to file.
        
        Args:
            tree: Tree to save
            output_path: Path to save to
        """
        self.version_manager.save_tree(tree, output_path)
