"""
Base classes for CRUD handlers.

Provides common data structures and base class for downstream task handlers
that process retrieved nodes with a single LLM call.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_default_client, TokenUsage, CompletionResult
from pipeline_execution.crud.crud_prompt_loader import PromptLoader


logger = logging.getLogger(__name__)


@dataclass
class SelectedNode:
    """
    A node selected as relevant by a handler.
    
    Attributes:
        tree_path: Full path to the node in the tree
        node_data: The node's data dictionary
        reasoning: Why this node was selected
    """
    tree_path: str
    node_data: Dict[str, Any]
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_path": self.tree_path,
            "node_data": self.node_data,
            "reasoning": self.reasoning
        }


@dataclass
class ReadResult:
    """Result specific to READ operations."""
    selected_nodes: List[SelectedNode] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_nodes": [n.to_dict() for n in self.selected_nodes]
        }


@dataclass
class DeleteResult:
    """Result specific to DELETE operations."""
    nodes_to_delete: List[str] = field(default_factory=list)  # tree paths
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes_to_delete": self.nodes_to_delete,
            "reasoning": self.reasoning
        }


@dataclass
class UpdateItem:
    """A single update to be applied."""
    tree_path: str
    updated_content: ET.Element
    original_content: Dict[str, Any]
    changes: Dict[str, Any]  # field -> {from: old, to: new}
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_path": self.tree_path,
            "changes": self.changes,
            "reasoning": self.reasoning
        }


@dataclass
class UpdateResult:
    """Result specific to UPDATE operations."""
    updates: List[UpdateItem] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "updates": [u.to_dict() for u in self.updates]
        }


@dataclass
class CreateResult:
    """Result specific to CREATE operations."""
    parent_path: str = ""  # Path to insert under
    position: int = -1  # Index among siblings (-1 for append)
    created_content: Optional[ET.Element] = None
    node_type: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_path": self.parent_path,
            "position": self.position,
            "node_type": self.node_type,
            "fields": self.fields,
            "reasoning": self.reasoning
        }


@dataclass
class HandlerResult:
    """
    Result from a downstream task handler.
    
    Attributes:
        success: Whether the operation succeeded
        operation: The CRUD operation type
        output: Operation-specific output (ReadResult, DeleteResult, etc.)
        token_usage: Token usage from the LLM call
        processing_time_ms: Time taken for the handler to process
        raw_response: Raw LLM response for debugging
        error: Error message if failed
    """
    success: bool
    operation: str
    output: Any = None  # ReadResult, DeleteResult, UpdateResult, or CreateResult
    token_usage: Optional[TokenUsage] = None
    processing_time_ms: float = 0.0
    raw_response: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "operation": self.operation,
            "processing_time_ms": self.processing_time_ms,
        }
        
        if self.output:
            result["output"] = self.output.to_dict()
        
        if self.token_usage:
            result["token_usage"] = self.token_usage.to_dict()
        
        if self.error:
            result["error"] = self.error
            
        return result


class BaseHandler(ABC):
    """
    Base class for downstream task handlers.
    
    Each handler processes retrieved nodes with a single LLM call
    to perform relevance reasoning and task-specific output generation.
    """
    
    DEFAULT_TRACES_PATH = Path(__file__).parent.parent.parent / "traces" / "reasoning_traces"
    
    def __init__(
        self,
        client=None,
        schema: Optional[Dict[str, Any]] = None,
        save_traces: bool = True,
        traces_path: Path = None,
        config: dict = None,
    ):
        """
        Initialize the handler.
        
        Args:
            client: Optional OpenAI client
            schema: Optional schema dict for node type information
            save_traces: Whether to save reasoning traces
            traces_path: Optional custom path for trace files. If None, traces are not saved.
            config: Optional config dict for client creation (avoids config.yaml fallback)
        """
        self._client = client
        self.schema = schema or {}
        self.save_traces = save_traces
        self._system_prompt = None
        self.traces_path = traces_path
        self._prompt_loader = None
        self._config = config
        
        # Only create traces directory if explicitly provided
        if self.traces_path:
            self.traces_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from client import OpenAIClient
            self._client = OpenAIClient(config=self._config)
        return self._client
    
    @property
    def prompt_loader(self) -> PromptLoader:
        """Lazy load the prompt loader."""
        if self._prompt_loader is None:
            self._prompt_loader = PromptLoader(schema=self.schema)
        return self._prompt_loader
    
    @property
    @abstractmethod
    def prompt_file(self) -> str:
        """Name of the prompt file for this handler."""
        pass
    
    @property
    def system_prompt(self) -> str:
        """Lazy load the system prompt via dynamic prompt composition."""
        if self._system_prompt is None:
            handler_type = self.prompt_file.replace(".txt", "")
            self._system_prompt = self.prompt_loader.load_prompt(handler_type)
        return self._system_prompt
    
    @abstractmethod
    def process(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        operation_context: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """
        Process retrieved nodes with a single LLM call.
        
        Args:
            user_query: The original user query
            retrieved_nodes: Nodes retrieved from semantic XPath execution
            operation_context: Additional context (e.g., parsed query info)
            
        Returns:
            HandlerResult with operation-specific output
        """
        pass
    
    def _format_nodes_for_prompt(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Format retrieved nodes for inclusion in the LLM prompt.
        
        Args:
            nodes: List of retrieved node dictionaries
            
        Returns:
            Formatted string representation of nodes
        """
        lines = []
        for i, node in enumerate(nodes):
            node_id = str(i + 1)
            tree_path = node.get("tree_path", f"node_{i}")
            node_data = node.get("node", {})
            node_type = node_data.get("type", "Unknown")
            score = node.get("score", 0.0)
            children = node.get("children", [])
            
            # Node header
            lines.append(f"[{node_id}] Path: {tree_path}")
            lines.append(f"    Type: {node_type}")
            lines.append(f"    Semantic Score: {score:.3f}")
            
            # Context / Reasoning Trace from Traversal
            context_trace = node.get("context_trace", [])
            if context_trace:
                lines.append("    Matched Context:")
                for item in context_trace:
                    lines.append(f"      - {item.get('description', '')}")
            
            # Node fields
            for field_name, field_value in node_data.items():
                if field_name in ("type", "children"):
                    continue
                if field_value:
                    if isinstance(field_value, list):
                        lines.append(f"    {field_name}: {', '.join(str(v) for v in field_value)}")
                    else:
                        display_value = str(field_value)[:200]
                        if len(str(field_value)) > 200:
                            display_value += "..."
                        lines.append(f"    {field_name}: {display_value}")
            
            # Full subtree with all fields
            if children:
                lines.append(f"    Subtree ({len(children)} children):")
                subtree_lines = self._format_subtree(children, indent=3)
                lines.extend(subtree_lines)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_subtree(
        self,
        children: List[Dict[str, Any]],
        indent: int = 2,
        max_depth: int = 3
    ) -> List[str]:
        """
        Recursively format a subtree with all fields and nested children.
        
        Args:
            children: List of child node dictionaries (may contain nested 'children')
            indent: Current indentation level (number of 2-space units)
            max_depth: Maximum recursion depth to prevent excessive output
            
        Returns:
            List of formatted lines
        """
        if max_depth <= 0:
            return []
        
        lines = []
        prefix = "  " * indent
        
        for child in children:
            child_type = child.get("type", "?")
            child_name = child.get("name", child.get("title", ""))
            
            # Node header with type and name
            if child_name:
                lines.append(f"{prefix}{child_type}: {child_name}")
            else:
                lines.append(f"{prefix}{child_type}")
            
            # Important fields (excluding type, children, attributes)
            field_prefix = prefix + "  "
            for field_name, field_value in child.items():
                if field_name in ("type", "children", "attributes", "name", "title"):
                    continue
                if field_value:
                    if isinstance(field_value, list):
                        # Truncate long lists
                        display_items = field_value[:5]
                        display_value = ", ".join(str(v) for v in display_items)
                        if len(field_value) > 5:
                            display_value += f"... (+{len(field_value) - 5} more)"
                        lines.append(f"{field_prefix}{field_name}: {display_value}")
                    else:
                        # Truncate long values
                        display_value = str(field_value)[:150]
                        if len(str(field_value)) > 150:
                            display_value += "..."
                        lines.append(f"{field_prefix}{field_name}: {display_value}")
            
            # Recursively format nested children
            nested_children = child.get("children", [])
            if nested_children:
                nested_lines = self._format_subtree(
                    nested_children,
                    indent=indent + 1,
                    max_depth=max_depth - 1
                )
                lines.extend(nested_lines)
        
        return lines
    
    def _make_llm_call(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> CompletionResult:
        """
        Make an LLM call with the handler's system prompt.
        
        Args:
            prompt: The user prompt
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            CompletionResult with response and token usage
        """
        return self.client.complete_with_usage(
            prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _save_trace(self, trace_data: Dict[str, Any], prefix: str):
        """Save a trace file for debugging."""
        if not self.save_traces or self.traces_path is None:
            return
            
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_file = self.traces_path / f"{prefix}_{timestamp}.json"
        
        import json
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved trace to {trace_file}")
