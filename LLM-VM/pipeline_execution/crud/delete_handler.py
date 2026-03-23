"""
Delete Handler - Downstream task handler for DELETE operations.

Uses a single LLM call to perform relevance reasoning and determine
which nodes should be deleted based on the user's request.

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
    DeleteResult
)
from pipeline_execution.crud.tree_modification_mixin import (
    TreeModificationMixin,
    ModificationResult
)


logger = logging.getLogger(__name__)


class DeleteHandler(BaseHandler, TreeModificationMixin):
    """
    Handler for DELETE operations.
    
    Takes retrieved nodes and uses a single LLM call to:
    1. Reason about which nodes match the deletion criteria
    2. Return the list of node paths to delete
    
    Also supports applying deletions to version content via the mixin.
    """
    
    @property
    def prompt_file(self) -> str:
        return "delete_handler.txt"
    
    def process(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        operation_context: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """
        Process DELETE operation with a single LLM call.
        
        Args:
            user_query: The original user query
            retrieved_nodes: Nodes retrieved from semantic XPath execution
            operation_context: Additional context (not used for DELETE)
            
        Returns:
            HandlerResult with DeleteResult containing nodes to delete
        """
        start_time = time.perf_counter()
        
        if not retrieved_nodes:
            return HandlerResult(
                success=True,
                operation="DELETE",
                output=DeleteResult(nodes_to_delete=[], reasoning="No candidates to delete"),
                processing_time_ms=0.0
            )
        
        # Format nodes for the prompt
        nodes_text = self._format_nodes_for_prompt(retrieved_nodes)
        
        # Format prompt
        context_str = ""
        if operation_context and operation_context.get("version_change_context"):
            context_str = f"Version Change Context: {operation_context.get('version_change_context')}\nUse this context to disambiguate which nodes are relevant.\n\n"

        prompt = f"""User Query: {user_query}

{context_str}Candidate Nodes:
{nodes_text}

Analyze each node and determine which ones should be deleted based on the user's request.
"""
        
        try:
            # Make single LLM call
            result = self._make_llm_call(prompt)
            
            # Parse response
            nodes_to_delete, reasoning = self._parse_response(result.content, retrieved_nodes)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Save trace
            if self.save_traces:
                self._save_trace({
                    "user_query": user_query,
                    "candidates_count": len(retrieved_nodes),
                    "nodes_to_delete": nodes_to_delete,
                    "reasoning": reasoning,
                    "raw_response": result.content,
                    "token_usage": result.usage.to_dict(),
                    "processing_time_ms": processing_time
                }, "delete_handler")
            
            return HandlerResult(
                success=True,
                operation="DELETE",
                output=DeleteResult(nodes_to_delete=nodes_to_delete, reasoning=reasoning),
                token_usage=result.usage,
                processing_time_ms=processing_time,
                raw_response=result.content
            )
            
        except Exception as e:
            logger.error(f"Error in DeleteHandler: {e}")
            processing_time = (time.perf_counter() - start_time) * 1000
            return HandlerResult(
                success=False,
                operation="DELETE",
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
        Apply deletions to version content.
        
        Args:
            handler_result: Result from process() containing nodes to delete
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
        
        delete_result = handler_result.output
        nodes_to_delete = delete_result.nodes_to_delete
        
        if not nodes_to_delete:
            return ModificationResult(
                success=False,
                error="No nodes selected for deletion"
            )
        
        return self.apply_deletions(
            version_content, 
            nodes_to_delete, 
            version,
            schema_name=self.schema.get("name"),
            config=self._config
        )
    
    def _parse_response(
        self,
        response: str,
        retrieved_nodes: List[Dict[str, Any]]
    ) -> tuple[List[str], str]:
        """Parse LLM response into list of paths to delete and reasoning."""
        nodes_to_delete = []
        overall_reasoning = ""
        
        try:
            # Find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                delete_list = parsed.get("nodes_to_delete", [])
                overall_reasoning = parsed.get("reasoning", "")
                
                for item in delete_list:
                    # Try to get path from response first
                    path = item.get("path", "")
                    
                    # If no path, look up by ID
                    if not path:
                        node_id = str(item.get("id", ""))
                        try:
                            idx = int(node_id) - 1
                            if 0 <= idx < len(retrieved_nodes):
                                path = retrieved_nodes[idx].get("tree_path", "")
                        except ValueError:
                            continue
                    
                    if path:
                        nodes_to_delete.append(path)
                
                return nodes_to_delete, overall_reasoning
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse delete handler response: {e}")
        
        # Return empty list on parse error (conservative)
        return [], "Parse error - no nodes selected for deletion"
