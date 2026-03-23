"""
Read Handler - Downstream task handler for READ operations.

Uses a single LLM call to perform relevance reasoning and select
the final nodes that match the user's query.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional

from pipeline_execution.crud.base import (
    BaseHandler,
    HandlerResult,
    ReadResult,
    SelectedNode
)


logger = logging.getLogger(__name__)


class ReadHandler(BaseHandler):
    """
    Handler for READ operations.
    
    Takes retrieved nodes and uses a single LLM call to:
    1. Reason about which nodes are relevant to the user's query
    2. Return the selected nodes with reasoning
    """
    
    @property
    def prompt_file(self) -> str:
        return "read_handler.txt"
    
    def process(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        operation_context: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """
        Process READ operation with a single LLM call.
        
        Args:
            user_query: The original user query
            retrieved_nodes: Nodes retrieved from semantic XPath execution
            operation_context: Additional context (not used for READ)
            
        Returns:
            HandlerResult with ReadResult containing selected nodes
        """
        start_time = time.perf_counter()
        
        if not retrieved_nodes:
            return HandlerResult(
                success=True,
                operation="READ",
                output=ReadResult(selected_nodes=[]),
                processing_time_ms=0.0
            )
        
        # Format nodes for the prompt
        nodes_text = self._format_nodes_for_prompt(retrieved_nodes)
        
        # Format prompt
        context_str = ""
        if operation_context and operation_context.get("version_change_context"):
            context_str = f"Version Change Context: {operation_context.get('version_change_context')}\nUse this context to disambiguate which nodes are relevant (e.g. if the user asks for a deleted item, look for the item mentioned in the context).\n\n"

        prompt = f"""User Query: {user_query}

{context_str}Candidate Nodes:
{nodes_text}

Analyze each node and determine which ones are relevant to the user's query.
"""
        
        try:
            # Make single LLM call
            result = self._make_llm_call(prompt)
            
            # Parse response
            selected_nodes = self._parse_response(result.content, retrieved_nodes)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Save trace
            if self.save_traces:
                self._save_trace({
                    "user_query": user_query,
                    "candidates_count": len(retrieved_nodes),
                    "selected_count": len(selected_nodes),
                    "selected_nodes": [n.to_dict() for n in selected_nodes],
                    "raw_response": result.content,
                    "token_usage": result.usage.to_dict(),
                    "processing_time_ms": processing_time
                }, "read_handler")
            
            return HandlerResult(
                success=True,
                operation="READ",
                output=ReadResult(selected_nodes=selected_nodes),
                token_usage=result.usage,
                processing_time_ms=processing_time,
                raw_response=result.content
            )
            
        except Exception as e:
            logger.error(f"Error in ReadHandler: {e}")
            processing_time = (time.perf_counter() - start_time) * 1000
            return HandlerResult(
                success=False,
                operation="READ",
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def _parse_response(
        self,
        response: str,
        retrieved_nodes: List[Dict[str, Any]]
    ) -> List[SelectedNode]:
        """Parse LLM response into SelectedNode objects."""
        selected = []
        
        try:
            # Find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                selected_list = parsed.get("selected_nodes", [])
                
                for item in selected_list:
                    node_id = str(item.get("id", ""))
                    try:
                        idx = int(node_id) - 1  # Convert to 0-based
                        if 0 <= idx < len(retrieved_nodes):
                            node = retrieved_nodes[idx]
                            selected.append(SelectedNode(
                                tree_path=node.get("tree_path", f"node_{idx}"),
                                node_data=node.get("node", {}),
                                reasoning=item.get("reasoning", "")
                            ))
                    except ValueError:
                        continue
                
                return selected
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse read handler response: {e}")
        
        # Fallback: return all nodes as selected
        logger.warning("Using fallback: selecting all nodes")
        for i, node in enumerate(retrieved_nodes):
            selected.append(SelectedNode(
                tree_path=node.get("tree_path", f"node_{i}"),
                node_data=node.get("node", {}),
                reasoning="Fallback: selected due to parse error"
            ))
        
        return selected
