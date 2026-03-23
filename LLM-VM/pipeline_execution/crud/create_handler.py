"""
Create Handler - Downstream task handler for CREATE operations.

Uses a single LLM call to determine insertion point and generate
new node content based on context.

Also handles tree modifications when provided with version content.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

from pipeline_execution.crud.base import (
    BaseHandler,
    HandlerResult,
    CreateResult
)
from pipeline_execution.crud.tree_modification_mixin import (
    TreeModificationMixin,
    ModificationResult
)


logger = logging.getLogger(__name__)


class CreateHandler(BaseHandler, TreeModificationMixin):
    """
    Handler for CREATE operations.
    
    Takes retrieved context nodes and uses a single LLM call to:
    1. Determine the best insertion point (parent path and position)
    2. Generate complete content for the new node
    
    Also supports applying insertions to version content via the mixin.
    """
    
    @property
    def prompt_file(self) -> str:
        return "create_handler.txt"
    
    def process(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        operation_context: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """
        Process CREATE operation with a single LLM call.
        
        Args:
            user_query: The original user query
            retrieved_nodes: Context nodes retrieved from semantic XPath execution
            operation_context: Additional context including node_type and description hints
            
        Returns:
            HandlerResult with CreateResult containing parent_path, position, and content
        """
        start_time = time.perf_counter()
        
        # Extract create info from context
        create_info = {}
        if operation_context:
            create_info = operation_context.get("create_info", {})
        
        node_type = create_info.get("node_type", "POI")
        description = create_info.get("description", "")
        
        # Format context nodes
        context_text = self._format_context_nodes(retrieved_nodes)
        
        # Build operation context section
        op_context_text = f"""
Operation Context:
- Node Type to Create: {node_type}
- Description: {description}
"""
        
        version_change_context = ""
        if operation_context and operation_context.get("version_change_context"):
            version_change_context = f"Version Change Context: {operation_context.get('version_change_context')}\nUse this context to inform content generation or placement if relevant.\n\n"
        
        prompt = f"""User Query: {user_query}
{op_context_text}
{version_change_context}Context Nodes (to help determine placement):
{context_text}

Determine the best insertion point and generate complete content for the new node.
"""
        
        try:
            # Make single LLM call with higher token limit for content generation
            result = self._make_llm_call(prompt, max_tokens=4096)
            
            # Parse response
            create_result = self._parse_response(result.content, node_type)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Save trace
            if self.save_traces:
                self._save_trace({
                    "user_query": user_query,
                    "context_nodes_count": len(retrieved_nodes),
                    "create_info": create_info,
                    "result": create_result.to_dict(),
                    "raw_response": result.content,
                    "token_usage": result.usage.to_dict(),
                    "processing_time_ms": processing_time
                }, "create_handler")
            
            return HandlerResult(
                success=create_result.created_content is not None,
                operation="CREATE",
                output=create_result,
                token_usage=result.usage,
                processing_time_ms=processing_time,
                raw_response=result.content
            )
            
        except Exception as e:
            logger.error(f"Error in CreateHandler: {e}")
            processing_time = (time.perf_counter() - start_time) * 1000
            return HandlerResult(
                success=False,
                operation="CREATE",
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def apply_to_content(
        self,
        handler_result: HandlerResult,
        version_content: List[ET.Element],
        version: ET.Element = None
    ) -> ModificationResult:
        """
        Apply creation to version content.
        
        Args:
            handler_result: Result from process() containing create info
            version_content: Copy of version content to modify
            version: Optional version element for path adjustment
            
        Returns:
            ModificationResult with modified content
        """
        if not handler_result.success or not handler_result.output:
            return ModificationResult(
                success=False,
                error=handler_result.error or "Handler processing failed"
            )
        
        create_result = handler_result.output
        
        if not create_result.created_content:
            return ModificationResult(
                success=False,
                error="Content generation failed"
            )
        
        return self.apply_insertion(
            version_content,
            create_result.parent_path,
            create_result.created_content,
            create_result.position,
            version,
            schema_name=self.schema.get("name"),
            config=self._config
        )
    
    def _format_context_nodes(self, nodes: List[Dict[str, Any]]) -> str:
        """Format context nodes with full subtree for placement context."""
        if not nodes:
            return "No context nodes available."
        
        lines = []
        for i, node in enumerate(nodes):
            node_id = str(i + 1)
            tree_path = node.get("tree_path", f"node_{i}")
            node_data = node.get("node", {})
            node_type = node_data.get("type", "Unknown")
            children = node.get("children", [])
            
            # Node header
            lines.append(f"[{node_id}] Path: {tree_path}")
            lines.append(f"    Type: {node_type}")
            
            # Key fields
            for field_name in ["name", "title", "description"]:
                if field_name in node_data and node_data[field_name]:
                    value = str(node_data[field_name])[:100]
                    lines.append(f"    {field_name}: {value}")
            
            # Full subtree with all fields for placement context
            if children:
                lines.append(f"    Subtree ({len(children)} children):")
                subtree_lines = self._format_subtree(children, indent=3)
                lines.extend(subtree_lines)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_response(self, response: str, default_node_type: str) -> CreateResult:
        """Parse LLM response into CreateResult."""
        try:
            # Find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                parent_path = parsed.get("parent_path", "")
                position = int(parsed.get("position", -1))
                node_type = parsed.get("node_type", default_node_type)
                fields = parsed.get("fields", {})
                reasoning = parsed.get("reasoning", "")
                
                # Create XML element
                xml_element = self._create_xml_element(node_type, fields)
                
                return CreateResult(
                    parent_path=parent_path,
                    position=position,
                    created_content=xml_element,
                    node_type=node_type,
                    fields=fields,
                    reasoning=reasoning
                )
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse create handler response: {e}")
        
        return CreateResult(
            parent_path="",
            position=-1,
            created_content=None,
            reasoning="Parse error - could not generate content"
        )
    
    def _create_xml_element(self, node_type: str, fields: Dict[str, Any]) -> ET.Element:
        """Create an XML element from fields."""
        element = ET.Element(node_type)
        
        for field_name, field_value in fields.items():
            if field_value is None:
                continue
            
            if isinstance(field_value, list):
                # Handle list fields like highlights
                container = ET.SubElement(element, field_name)
                item_name = field_name.rstrip("s")  # "highlights" -> "highlight"
                for item in field_value:
                    child = ET.SubElement(container, item_name)
                    child.text = str(item)
            else:
                child = ET.SubElement(element, field_name)
                child.text = str(field_value)
        
        return element
