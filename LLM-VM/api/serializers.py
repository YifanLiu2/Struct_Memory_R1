"""
Shared JSON serializers for the Semantic XPath pipeline.

Used by both:
- Direct pipeline access (eval mode) for JSON trace output
- Flask API (demo mode) for HTTP responses

Provides consistent JSON representation of:
- XML trees with optional score annotations
- Pipeline execution results
- Traversal steps with scoring details
- Score fusion traces
"""

import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List


def build_tree_path(element: ET.Element, parent_path: str = "") -> str:
    """
    Build a human-readable tree path for an element.
    
    IMPORTANT: This must match the format used by NodeUtils.get_name() in the
    execution engine, so that paths from traversal steps match tree paths.
    
    Args:
        element: The XML element
        parent_path: Path of the parent element
        
    Returns:
        Path string like "Itinerary > Day 1 > CN Tower"
    """
    # Get the element's display name - must match NodeUtils.get_name() format!
    # For container nodes with index or number, use "{Tag} {index}" format
    index = element.get("index") or element.get("number")
    if index is not None:
        display = f"{element.tag} {index}"
    else:
        # For leaf nodes, try common name fields
        name_elem = element.find("name")
        if name_elem is not None and name_elem.text:
            display = name_elem.text  # Just the name, not "Tag: Name"
        else:
            # Try other common name fields
            for field in ("title", "label"):
                name_elem = element.find(field)
                if name_elem is not None and name_elem.text:
                    display = name_elem.text
                    break
            else:
                # Fallback to tag name
                display = element.tag
    
    if parent_path:
        return f"{parent_path} > {display}"
    return display


def serialize_tree(
    root: ET.Element, 
    scores: Optional[Dict[str, float]] = None,
    parent_path: str = ""
) -> Dict[str, Any]:
    """
    Convert XML tree to JSON with optional score annotations.
    
    Args:
        root: The XML element to serialize
        scores: Optional dict mapping tree paths to scores
        parent_path: Path of parent for building full paths
        
    Returns:
        JSON-serializable dictionary representing the tree
    """
    scores = scores or {}
    current_path = build_tree_path(root, parent_path)
    
    # Separate fields (leaf text nodes) from children (container nodes)
    fields = {}
    children = []
    
    for child in root:
        if len(list(child)) == 0 and child.text:
            # Leaf node with text - it's a field
            fields[child.tag] = child.text
        elif child.tag == "highlights":
            # Handle highlights as a list
            highlights = [h.text for h in child.findall("highlight") if h.text]
            fields["highlights"] = highlights
        else:
            # Container node - recurse
            children.append(serialize_tree(child, scores, current_path))
    
    result = {
        "tag": root.tag,
        "path": current_path,
        "attributes": dict(root.attrib) if root.attrib else {},
    }
    
    if fields:
        result["fields"] = fields
    
    if children:
        result["children"] = children
    
    # Add score if available
    score = scores.get(current_path)
    if score is not None:
        result["score"] = round(score, 4)
    
    return result


def serialize_node(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize a matched node from pipeline results.
    """
    result = {
        "path": node_data.get("tree_path", ""),
        "score": round(node_data.get("score", 0), 4),
        "type": node_data.get("node", {}).get("type", ""),
        "name": node_data.get("node", {}).get("name", ""),
    }
    
    # Add optional fields if present
    node = node_data.get("node", {})
    if node.get("description"):
        result["description"] = node["description"]
    if node.get("time_block"):
        result["timeBlock"] = node["time_block"]
    if node.get("expected_cost"):
        result["expectedCost"] = node["expected_cost"]
    if node.get("highlights"):
        result["highlights"] = node["highlights"]
    
    return result


def serialize_timing(timing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize timing information from pipeline execution.
    """
    steps = timing.get("steps", [])
    return {
        "steps": [
            {
                "step": step.get("step", ""),
                "timeMs": round(step.get("time_ms", 0), 1)
            }
            for step in steps
        ],
        "totalMs": round(timing.get("total_ms", 0), 1)
    }


def serialize_scoring_result(scoring_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize scoring result with full nested structure for visualization.
    
    The scoring_steps structure is kept as-is (snake_case) since it's a complex
    nested structure that the frontend handles directly.
    
    Args:
        scoring_result: Scoring result from predicate scorer
        
    Returns:
        Formatted scoring result for frontend
    """
    if not scoring_result:
        return {}
    
    result = {
        "predicate": scoring_result.get("predicate", ""),
        "predicateAst": scoring_result.get("predicate_ast", {}),
        "config": scoring_result.get("config", {}),
        "nodeScores": [],
        "ranking": scoring_result.get("ranking", []),
    }
    
    # Serialize node scores with their scoring steps
    # Keep scoring_steps as-is since it has complex nested structure
    for node_score in scoring_result.get("node_scores", []):
        result["nodeScores"].append({
            "nodeIdx": node_score.get("node_idx", 0),
            "nodeName": node_score.get("node_name", ""),
            "nodeType": node_score.get("node_type", ""),
            "scoringSteps": node_score.get("scoring_steps", []),  # Keep nested structure as-is
            "finalScore": round(node_score.get("final_score", 0), 4),
        })
    
    # Add batch scoring info if available
    if "batch_scoring" in scoring_result:
        batch = scoring_result["batch_scoring"]
        result["batchScoring"] = {
            "semanticValues": batch.get("semantic_values", []),
            "perValueStats": batch.get("per_value_stats", []),
            "totalScorerCalls": batch.get("total_scorer_calls", 0),
            "totalDescriptionsScored": batch.get("total_descriptions_scored", 0),
        }
    
    # Add token usage if available
    if "token_usage" in scoring_result and scoring_result["token_usage"]:
        result["tokenUsage"] = scoring_result["token_usage"]
    
    return result


def serialize_traversal_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize a single traversal step for visualization.
    
    Args:
        step: TraversalStep.to_dict() data or raw dict
        
    Returns:
        Formatted step data for frontend with camelCase keys
    """
    # Handle both TraversalStep.to_dict() output and raw dicts
    if hasattr(step, 'to_dict'):
        step = step.to_dict()
    
    details = step.get("details", {})
    
    # Serialize details with proper camelCase conversion
    serialized_details = {}
    if details:
        # Handle scoring_result specially
        if "scoring_result" in details and details["scoring_result"]:
            serialized_details["scoringResult"] = serialize_scoring_result(details["scoring_result"])
        
        # Convert other detail fields to camelCase
        field_mapping = {
            "predicate": "predicate",
            "predicate_type": "predicateType",
            "target_type": "targetType",
            "found_count": "foundCount",
            "index": "index",
            "selected_count": "selectedCount",
            "groups_processed": "groupsProcessed",
            "group_details": "groupDetails",
            "matched": "matched",
        }
        
        for snake_key, camel_key in field_mapping.items():
            if snake_key in details:
                serialized_details[camel_key] = details[snake_key]
    
    return {
        "stepIndex": step.get("step_index", 0),
        "stepQuery": step.get("step_query", ""),
        "action": step.get("action", ""),
        "nodesBeforeCount": step.get("nodes_before_count", 0),
        "nodesAfterCount": step.get("nodes_after_count", 0),
        "nodesBefore": step.get("nodes_before", []),
        "nodesAfter": step.get("nodes_after", []),
        "details": serialized_details
    }


def serialize_score_fusion(fusion_trace: Any) -> Dict[str, Any]:
    """
    Serialize score fusion trace for visualization.
    
    Args:
        fusion_trace: ScoreFusionTrace object or dict
        
    Returns:
        Formatted fusion data showing per-node score breakdown
    """
    if fusion_trace is None:
        return {"perNode": []}
    
    # Handle both object and dict
    if hasattr(fusion_trace, 'to_dict'):
        fusion_trace = fusion_trace.to_dict()
    
    per_node = fusion_trace.get("per_node", [])
    return {
        "perNode": [
            {
                "path": node.get("node_path", ""),
                "type": node.get("node_type", ""),
                "stepContributions": [
                    {
                        "stepIndex": contrib.get("step_index", 0),
                        "predicate": contrib.get("predicate", ""),
                        "score": round(contrib.get("score", 0), 4)
                    }
                    for contrib in node.get("step_contributions", [])
                ],
                "accumulatedProduct": round(node.get("accumulated_product", 1.0), 4),
                "finalScore": round(node.get("final_score", 0), 4)
            }
            for node in per_node
        ]
    }


def serialize_final_filtering(filtering: Any) -> Dict[str, Any]:
    """
    Serialize final filtering trace for visualization.
    
    Args:
        filtering: FinalFilteringTrace object or dict
        
    Returns:
        Formatted filtering data for frontend
    """
    if filtering is None:
        return {}
    
    # Handle both object and dict
    if hasattr(filtering, 'to_dict'):
        filtering = filtering.to_dict()
    
    filtered_nodes = []
    for node in filtering.get("filtered_nodes", []):
        entry = {
            "path": node.get("path", ""),
            "score": round(node.get("score", 0), 4),
            "type": node.get("type", ""),
        }
        # Include own/parent score breakdown when parent propagation was applied
        if "own_score" in node:
            entry["ownScore"] = round(node["own_score"], 4)
        if "parent_score" in node and node["parent_score"] != 1.0:
            entry["parentScore"] = round(node["parent_score"], 4)
        filtered_nodes.append(entry)
    
    return {
        "beforeFilterCount": filtering.get("before_filter_count", 0),
        "threshold": filtering.get("threshold", 0),
        "topK": filtering.get("top_k", 0),
        "afterFilterCount": filtering.get("after_filter_count", 0),
        "filteredNodes": filtered_nodes,
    }


def serialize_demo_logger_trace(demo_trace: Any) -> Dict[str, Any]:
    """
    Serialize demo logger trace for visualization.
    
    Args:
        demo_trace: DemoLoggerTrace object or dict
        
    Returns:
        Formatted demo logger data showing:
        - Per-step accumulated scores
        - Child-to-parent score contributions
    """
    if demo_trace is None:
        return {"stepTraces": [], "accumulatedScores": {}}
    
    # Handle both object and dict
    if hasattr(demo_trace, 'to_dict'):
        demo_trace = demo_trace.to_dict()
    
    # Serialize step traces with accumulated scores
    step_traces = demo_trace.get("step_traces", [])
    
    return {
        "stepTraces": step_traces,  # Already in correct format from demo_logger
        "accumulatedScores": {
            path: round(score, 4) 
            for path, score in demo_trace.get("accumulated_scores", {}).items()
        }
    }


def serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize full pipeline result for API response.
    
    Args:
        result: Result dict from SemanticXPathPipeline.process_request()
        
    Returns:
        JSON-serializable dict for API response
    """
    response = {
        "success": result.get("success", False),
        "operation": result.get("operation", "UNKNOWN"),
        "fullQuery": result.get("full_query", ""),
        "userQuery": result.get("user_query", ""),
        "xpathQuery": result.get("xpath_query", ""),
        "timestamp": result.get("timestamp", ""),
    }
    
    # Add timing
    if "timing" in result:
        response["timing"] = serialize_timing(result["timing"])
    
    # Add traversal steps for step-by-step visualization
    if "traversal_steps" in result and result["traversal_steps"]:
        response["traversalSteps"] = [
            serialize_traversal_step(step.to_dict() if hasattr(step, 'to_dict') else step)
            for step in result["traversal_steps"]
        ]
    
    # Add score fusion trace for understanding how scores are combined
    if "score_fusion_trace" in result and result["score_fusion_trace"]:
        response["scoreFusion"] = serialize_score_fusion(result["score_fusion_trace"])
    
    # Add final filtering trace
    if "final_filtering_trace" in result and result["final_filtering_trace"]:
        response["finalFiltering"] = serialize_final_filtering(result["final_filtering_trace"])
    
    # Add demo logger trace for detailed score visualization
    if "demo_logger_trace" in result and result["demo_logger_trace"]:
        response["demoLoggerTrace"] = serialize_demo_logger_trace(result["demo_logger_trace"])
    
    # Add intent classification
    if "intent" in result:
        intent = result["intent"]
        response["intent"] = {
            "type": intent.get("intent", ""),
            "confidence": intent.get("confidence", 0),
            "xpathHint": intent.get("xpath_hint", ""),
            "operationDetails": intent.get("operation_details", {})
        }
    
    # Add execution details
    execution = {}
    
    # Candidates and selected nodes
    if "candidates_count" in result:
        execution["candidatesCount"] = result["candidates_count"]
    if "selected_count" in result:
        execution["selectedCount"] = result["selected_count"]
    if "selected_nodes" in result:
        execution["selectedNodes"] = [
            serialize_node({"tree_path": n.get("tree_path", ""), "score": n.get("score", 1.0), "node": n})
            if "tree_path" not in n else serialize_node(n)
            for n in result["selected_nodes"]
        ]
    
    # Reasoning trace
    if "reasoning_trace" in result:
        trace = result["reasoning_trace"]
        execution["reasoning"] = {
            "selectedCount": trace.get("selected_count", 0),
            "rejectedCount": trace.get("rejected_count", 0),
            "decisions": trace.get("decisions", [])
        }
    
    if execution:
        response["execution"] = execution
    
    # Add operation-specific results
    if result.get("operation") == "DELETE":
        response["modification"] = {
            "deletedCount": result.get("deleted_count", 0),
            "deletedPaths": result.get("deleted_paths", [])
        }
    elif result.get("operation") == "UPDATE":
        response["modification"] = {
            "updatedCount": result.get("updated_count", 0),
            "updatedPaths": result.get("updated_paths", []),
            "updateResults": result.get("update_results", [])
        }
    elif result.get("operation") == "CREATE":
        response["modification"] = {
            "createdPath": result.get("created_path"),
            "insertionPoint": result.get("insertion_point", {}),
            "contentResult": result.get("content_result", {})
        }
    
    # Add tree version info
    if "tree_version" in result and result["tree_version"]:
        tv = result["tree_version"]
        response["treeVersion"] = {
            "version": tv.get("version"),
            "path": tv.get("path", ""),
            "operation": tv.get("operation", ""),
            "timestamp": tv.get("timestamp", "")
        }
    
    return response


def serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize pipeline configuration for API response.
    """
    executor_config = config.get("xpath_executor", {})
    return {
        "activeSchema": config.get("active_schema"),
        "activeData": config.get("active_data", ""),
        "mode": config.get("mode", "demo"),
        "executor": {
            "topK": executor_config.get("top_k", 5),
            "scoreThreshold": executor_config.get("score_threshold", 0.5),
            "scoringMethod": executor_config.get("scoring_method", "entailment")
        }
    }
