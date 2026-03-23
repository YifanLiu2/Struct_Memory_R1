"""
Update Handler - Downstream task handler for UPDATE operations.

Uses a single LLM call to perform relevance reasoning and generate
updated content for matching nodes.

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
    UpdateResult,
    UpdateItem
)
from pipeline_execution.crud.tree_modification_mixin import (
    TreeModificationMixin,
    ModificationResult
)


logger = logging.getLogger(__name__)


class UpdateHandler(BaseHandler, TreeModificationMixin):
    """
    Handler for UPDATE operations.
    
    Takes retrieved nodes and uses a single LLM call to:
    1. Reason about which nodes should be updated
    2. Generate updated content for each selected node
    
    Also supports applying updates to version content via the mixin.
    """
    
    @property
    def prompt_file(self) -> str:
        return "update_handler.txt"
    
    def process(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        operation_context: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """
        Process UPDATE operation with a single LLM call.
        
        Args:
            user_query: The original user query
            retrieved_nodes: Nodes retrieved from semantic XPath execution
            operation_context: Additional context (e.g., update hints from parsed query)
            
        Returns:
            HandlerResult with UpdateResult containing updates to apply
        """
        start_time = time.perf_counter()
        
        if not retrieved_nodes:
            return HandlerResult(
                success=True,
                operation="UPDATE",
                output=UpdateResult(updates=[]),
                processing_time_ms=0.0
            )
        
        # Format nodes for the prompt with full content
        nodes_text = self._format_nodes_with_content(retrieved_nodes)
        
        # Add operation context if available
        context_text = ""
        if operation_context:
            update_info = operation_context.get("update_info", {})
            if update_info:
                context_text += f"\nUpdate Hints: {json.dumps(update_info)}\n"
            
            if operation_context.get("version_change_context"):
                context_text += f"\nVersion Change Context: {operation_context.get('version_change_context')}\nUse this context to disambiguate which nodes are relevant.\n"
        
        prompt = f"""User Query: {user_query}
{context_text}
Candidate Nodes:
{nodes_text}

Analyze each node and determine which ones should be updated. For each update, provide the complete updated field set.
"""
        
        try:
            # Make single LLM call with higher token limit for content generation
            result = self._make_llm_call(prompt, max_tokens=4096)
            
            # Parse response
            updates = self._parse_response(result.content, retrieved_nodes)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Save trace
            if self.save_traces:
                self._save_trace({
                    "user_query": user_query,
                    "candidates_count": len(retrieved_nodes),
                    "updates_count": len(updates),
                    "updates": [u.to_dict() for u in updates],
                    "raw_response": result.content,
                    "token_usage": result.usage.to_dict(),
                    "processing_time_ms": processing_time
                }, "update_handler")
            
            return HandlerResult(
                success=True,
                operation="UPDATE",
                output=UpdateResult(updates=updates),
                token_usage=result.usage,
                processing_time_ms=processing_time,
                raw_response=result.content
            )
            
        except Exception as e:
            logger.error(f"Error in UpdateHandler: {e}")
            processing_time = (time.perf_counter() - start_time) * 1000
            return HandlerResult(
                success=False,
                operation="UPDATE",
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
        Apply updates to version content.
        
        Args:
            handler_result: Result from process() containing updates
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
        
        update_result = handler_result.output
        updates = update_result.updates
        
        if not updates:
            return ModificationResult(
                success=False,
                error="No nodes selected for update"
            )
        
        # Convert UpdateItem list to the format expected by apply_updates
        update_dicts = [
            {
                "tree_path": item.tree_path,
                "updated_content": item.updated_content
            }
            for item in updates
        ]
        
        return self.apply_updates(
            version_content,
            update_dicts,
            version,
            schema_name=self.schema.get("name"),
            config=self._config
        )
    
    def _format_nodes_with_content(self, nodes: List[Dict[str, Any]]) -> str:
        """Format nodes with full content for update context."""
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
            
            # All fields
            for field_name, field_value in node_data.items():
                if field_name in ("type", "children"):
                    continue
                if field_value:
                    if isinstance(field_value, list):
                        lines.append(f"    {field_name}: {', '.join(str(v) for v in field_value)}")
                    else:
                        lines.append(f"    {field_name}: {field_value}")
            
            # Full subtree with all fields
            if children:
                lines.append(f"    Subtree ({len(children)} children):")
                subtree_lines = self._format_subtree(children, indent=3)
                lines.extend(subtree_lines)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_response(
        self,
        response: str,
        retrieved_nodes: List[Dict[str, Any]]
    ) -> List[UpdateItem]:
        """Parse LLM response into UpdateItem objects."""
        updates = []
        
        try:
            # Find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                update_list = parsed.get("updates", [])
                
                for item in update_list:
                    node_id = str(item.get("id", ""))
                    try:
                        idx = int(node_id) - 1
                        if 0 <= idx < len(retrieved_nodes):
                            original_node = retrieved_nodes[idx]
                            original_data = original_node.get("node", {})
                            tree_path = item.get("path", original_node.get("tree_path", ""))
                            
                            fields = item.get("fields", {})
                            new_type = item.get("new_type")
                            reasoning = item.get("reasoning", "")
                            
                            # Determine node type
                            node_type = new_type if new_type else original_data.get("type", "POI")
                            
                            # Create XML element
                            xml_element = self._create_xml_element(node_type, fields)
                            
                            # Calculate changes
                            changes = self._calculate_changes(original_data, fields)
                            if new_type:
                                changes["type"] = {"from": original_data.get("type"), "to": new_type}
                            
                            updates.append(UpdateItem(
                                tree_path=tree_path,
                                updated_content=xml_element,
                                original_content=original_data,
                                changes=changes,
                                reasoning=reasoning
                            ))
                    except ValueError:
                        continue
                
                return updates
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse update handler response: {e}")
        
        return []
    
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
    
    def _calculate_changes(
        self,
        original: Dict[str, Any],
        updated: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate the changes between original and updated fields."""
        changes = {}
        
        all_fields = set(original.keys()) | set(updated.keys())
        skip_fields = {"type", "children", "attributes"}
        
        for field in all_fields:
            if field in skip_fields:
                continue
            
            old_val = original.get(field)
            new_val = updated.get(field)
            
            # Normalize for comparison
            if isinstance(old_val, list):
                old_val = sorted([str(v) for v in old_val]) if old_val else []
            if isinstance(new_val, list):
                new_val = sorted([str(v) for v in new_val]) if new_val else []
            
            if old_val != new_val:
                changes[field] = {"from": original.get(field), "to": updated.get(field)}
        
        return changes
