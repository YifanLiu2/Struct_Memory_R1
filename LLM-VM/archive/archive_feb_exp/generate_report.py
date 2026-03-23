import argparse
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Any
import yaml

def load_result_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_token_usage(usage: Dict[str, int]) -> str:
    if not usage:
        return "-"
    p = usage.get('prompt_tokens', 0)
    c = usage.get('completion_tokens', 0)
    t = usage.get('total_tokens', 0)
    return f"{t:,} ({p:,} / {c:,})"

def format_node_details(node: Dict[str, Any], indent: str = "") -> List[str]:
    """Format a node's details as markdown lines."""
    lines = []
    if "name" in node:
        lines.append(f"{indent}- **Name:** {node['name']}")
    if "type" in node:
        lines.append(f"{indent}- **Type:** {node['type']}")
    elif "node_type" in node:
        lines.append(f"{indent}- **Type:** {node['node_type']}")
    if "time_block" in node:
        lines.append(f"{indent}- **Time:** {node['time_block']}")
    if "expected_cost" in node:
        lines.append(f"{indent}- **Cost:** {node['expected_cost']}")
    if "description" in node:
        lines.append(f"{indent}- **Description:** {node['description']}")
    if "highlights" in node and node["highlights"]:
        highlights_str = ", ".join(node["highlights"])
        lines.append(f"{indent}- **Highlights:** {highlights_str}")
    if "reasoning" in node:
        lines.append(f"{indent}- **Reasoning:** {node['reasoning']}")
    if "tree_path" in node:
        lines.append(f"{indent}- **Path:** `{node['tree_path']}`")
    elif "path" in node:
        lines.append(f"{indent}- **Path:** `{node['path']}`")
    return lines



def format_incontext_stats(base_dir: Path, pipeline: str) -> List[str]:
    """Format simple stats table for incontext pipeline."""
    lines = []
    pipeline_dir = base_dir / pipeline
    
    if not pipeline_dir.exists():
        return lines
    
    query_count = 0
    total_time_ms = 0
    total_tokens = 0
    
    # Try both new folder structure and legacy flat structure
    result_files = sorted(glob.glob(str(pipeline_dir / "query_*" / "result.json")))
    if not result_files:
        result_files = sorted(glob.glob(str(pipeline_dir / "query_*_result.json")))

    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract top-level stats
            # Fallback to 0 if missing
            t_ms = data.get("execution_time_ms", 0)
            
            usage = data.get("token_usage", {})
            tokens = usage.get("total_tokens", 0)
            
            # Simple validation: if 0, try raw_result
            if t_ms == 0:
                t_ms = data.get("raw_result", {}).get("execution_time_ms", 0)
            if tokens == 0:
                tokens = data.get("raw_result", {}).get("token_usage", {}).get("total_tokens", 0)
            
            total_time_ms += t_ms
            total_tokens += tokens
            query_count += 1
            
        except Exception:
            continue
            
    if query_count == 0:
        return lines

    avg_time_s = (total_time_ms / query_count) / 1000
    avg_tokens = total_tokens / query_count
    total_time_s = total_time_ms / 1000
    
    lines.append(f"### Performance Summary ({query_count} queries)")
    lines.append("")
    lines.append("| Metric | Total | Average per Query |")
    lines.append("|---|---|---|")
    lines.append(f"| Time | {total_time_s:.2f}s | {avg_time_s:.2f}s |")
    lines.append(f"| Tokens | {total_tokens:,.0f} | {avg_tokens:,.0f} |")
    lines.append("")
    
    return lines


def format_stage_breakdown(base_dir: Path, pipeline: str) -> List[str]:
    """Format stage-by-stage timing and token breakdown for a pipeline."""
    lines = []
    pipeline_dir = base_dir / pipeline
    
    if not pipeline_dir.exists():
        return lines
    
    # Collect stage data from all result.json files
    all_stages = {}
    query_count = 0
    
    result_files = sorted(glob.glob(str(pipeline_dir / "query_*" / "result.json")))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_result = data.get("raw_result", {})
            timing = raw_result.get("timing", {})
            stages = timing.get("stages", [])
            
            # Fallback for pipelines without detailed stages (e.g. incontext)
            if not stages and "execution_time_ms" in data:
                 stages = [{
                     "name": "pipeline_execution",
                     "time_ms": data.get("execution_time_ms", 0),
                     "token_usage": data.get("token_usage", {})
                 }]

            if not stages:
                continue
            
            query_count += 1
            
            for stage in stages:
                name = stage.get("name", "unknown")
                time_ms = stage.get("time_ms", 0)
                token_usage = stage.get("token_usage", {})
                
                if name not in all_stages:
                    all_stages[name] = {
                        "total_time_ms": 0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens": 0,
                        "count": 0
                    }
                
                all_stages[name]["total_time_ms"] += time_ms
                all_stages[name]["count"] += 1
                if token_usage:
                    all_stages[name]["total_prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                    all_stages[name]["total_completion_tokens"] += token_usage.get("completion_tokens", 0)
                    all_stages[name]["total_tokens"] += token_usage.get("total_tokens", 0)
        except Exception:
            continue
    
    if not all_stages or query_count == 0:
        return lines
    
    # Calculate totals
    total_time = sum(s["total_time_ms"] for s in all_stages.values())
    total_tokens = sum(s["total_tokens"] for s in all_stages.values())
    
    # Stage order
    stage_order = ["version_resolution", "version_lookup", "query_generation", "xpath_execution", "downstream_task", "pipeline_execution"]
    
    lines.append(f"### Stage Breakdown ({query_count} queries)")
    lines.append("")
    lines.append("| Stage | Time (s) | Time % | Prompt | Completion | Total Tokens |")
    lines.append("|-------|----------|--------|--------|------------|--------------|")
    
    for stage_name in stage_order:
        if stage_name not in all_stages:
            continue
        stage = all_stages[stage_name]
        time_s = stage["total_time_ms"] / 1000
        time_pct = (stage["total_time_ms"] / total_time * 100) if total_time > 0 else 0
        prompt = stage["total_prompt_tokens"]
        completion = stage["total_completion_tokens"]
        tokens = stage["total_tokens"]
        
        lines.append(f"| {stage_name} | {time_s:.1f}s | {time_pct:.1f}% | {prompt:,} | {completion:,} | {tokens:,} |")
    
    # Total row
    total_prompt = sum(s["total_prompt_tokens"] for s in all_stages.values())
    total_completion = sum(s["total_completion_tokens"] for s in all_stages.values())
    lines.append(f"| **TOTAL** | **{total_time/1000:.1f}s** | **100%** | **{total_prompt:,}** | **{total_completion:,}** | **{total_tokens:,}** |")
    lines.append("")
    
    # Averages per query
    lines.append("**Averages per query:**")
    lines.append("")
    lines.append("| Stage | Avg Time | Avg Tokens |")
    lines.append("|-------|----------|------------|")
    
    for stage_name in stage_order:
        if stage_name not in all_stages:
            continue
        stage = all_stages[stage_name]
        stage_count = stage["count"]
        if stage_count == 0:
            continue
        avg_time = stage["total_time_ms"] / stage_count / 1000
        avg_tokens = stage["total_tokens"] / stage_count
        
        lines.append(f"| {stage_name} | {avg_time:.2f}s | {avg_tokens:,.0f} |")
    
    lines.append(f"| **TOTAL** | **{total_time/query_count/1000:.2f}s** | **{total_tokens/query_count:,.0f}** |")
    lines.append("")
    
    return lines


def format_scoring_table(execution_trace: Dict[str, Any]) -> List[str]:
    """Format scoring details as a markdown table."""
    lines = []
    
    # 1. Find the scoring result (try robust paths)
    scoring_result = None
    
    # Path A: explicit scoring_traces list (execution report)
    if "scoring_traces" in execution_trace and execution_trace["scoring_traces"]:
        scoring_result = execution_trace["scoring_traces"][0]
    
    # Path B: traversal steps
    elif "traversal_steps" in execution_trace:
        steps = execution_trace["traversal_steps"]
        if steps and "details" in steps[-1] and "scoring_result" in steps[-1]["details"]:
            scoring_result = steps[-1]["details"]["scoring_result"]
            
    if not scoring_result:
        return lines

    # 2. Extract key metadata
    predicate = scoring_result.get("predicate", "N/A")
    threshold = scoring_result.get("config", {}).get("score_threshold", 0.0)
    top_k = scoring_result.get("config", {}).get("top_k", 10)
    
    # Get the list of nodes that actually made it through top-k filtering
    final_filtering = execution_trace.get("final_filtering", {})
    filtered_nodes = final_filtering.get("filtered_nodes", [])
    filtered_node_paths = set()
    for fn in filtered_nodes:
        # Extract just the node name from path (e.g., "Day 5" from "Root > Itinerary_Version 1 > Itinerary > Day 5")
        path = fn.get("path", "")
        if " > " in path:
            node_name = path.split(" > ")[-1]
            filtered_node_paths.add(node_name)
    
    semantic_values = []
    if "batch_scoring" in scoring_result and "semantic_values" in scoring_result["batch_scoring"]:
        semantic_values = scoring_result["batch_scoring"]["semantic_values"]
    
    # 3. Build Table Header
    # Columns: Node Name | C1 (Value1) | C2 (Value2) | ... | Final Score | Result
    
    node_scores = scoring_result.get("node_scores", [])
    if not node_scores:
        return lines
        
    # Analyze scoring structure to determine columns
    # We want to drill down to find atomic predicates if possible
    first_node = node_scores[0]
    
    first_node = node_scores[0]
    
    # helper: recursively find atomic predicates in trace structure
    def find_atomic_columns(step, context=""):
        cols = []
        step_type = step.get("type", "")
        
        if step_type == "atom":
            # Found one!
            val = step.get("condition", {}).get("value", "unknown")
            if context:
                return [f"{val} ({context})"]
            return [val]
            
        elif step_type == "not":
            inner = step.get("inner_trace", [])
            if inner:
                return find_atomic_columns(inner[0], context)
            
        elif step_type == "agg_exists_recursive":
            # Update context with child type if available
            new_context = context
            if "child_type" in step:
                # Use strict type (ROI, Restaurant) or keep existing if nested
                new_context = step["child_type"]
            
            # Check the trace of the best match
            best_match = step.get("best_match_details")
            if best_match and "trace" in best_match and best_match["trace"]:
                return find_atomic_columns(best_match["trace"][0], new_context)
            # If no best match details (e.g. empty set), we might still want to know what columns *would* be there
            # But without a trace we can't be sure of the structure. 
            # Fallback: check child_results if any exist (they might have traces)
            # For now, if we can't find structure, we return empty.
                
        elif step_type in ("or", "and"):
            # Check inner traces for children
            inner_traces = step.get("inner_traces", [])
            for trace in inner_traces:
                if trace:
                    cols.extend(find_atomic_columns(trace[0], context))
            # If no inner traces but child scores exist (old trace format)
            if not cols and "child_scores" in step:
                 return [f"C{i+1}" for i in range(len(step["child_scores"]))]
                 
        return cols

    # helper: extract scores for those columns
    def extract_atomic_scores(step):
        scores = []
        step_type = step.get("type", "")
        
        if step_type == "atom":
            return [step.get("score", 0.0)]
            
        elif step_type == "not":
            inner = step.get("inner_trace", [])
            if inner:
                return extract_atomic_scores(inner[0])
            
        elif step_type == "agg_exists_recursive":
            best_match = step.get("best_match_details")
            if best_match and "trace" in best_match and best_match["trace"]:
                return extract_atomic_scores(best_match["trace"][0])
            # If no trace (e.g. no children found), we need to pad with 0.0s 
            # BUT we don't know how many columns are missing.
            # This is tricky. simpler to return empty and let the row padder handle it with 0.0s
            # provided the column discovery found the columns from a "full" node.
                
        elif step_type in ("or", "and"):
            inner_traces = step.get("inner_traces", [])
            for trace in inner_traces:
                if trace:
                    scores.extend(extract_atomic_scores(trace[0]))
                else:
                    scores.append(0.0) # We might insert just one 0.0, but if that branch had multiple cols, alignment breaks. 
                    # Ideally we need the schema of columns to pad correctly.
                    # For now, let's assume 1-to-1 or that alignment drifts are acceptable/fixable by context.
            if not scores and "child_scores" in step:
                return step.get("child_scores", [])
                
        return scores

    # Attempt to find columns from first node's trace
    data_columns = []
    # Find a node that has a "full" trace (meaning it found children) to determine column structure
    # If the first node has empty children (no POIs), it won't reveal the "POI: work" column structure.
    # So scan nodes until we find a good representative.
    for node in node_scores:
        if node.get("scoring_steps"):
            cols = find_atomic_columns(node["scoring_steps"][0])
            if len(cols) > len(data_columns):
                 data_columns = cols
    
    # If discovery still empty (e.g. all empty), fall back
    if not data_columns:
        if semantic_values:
            data_columns = semantic_values
        else:
            data_columns = ["Score"]

    header_cols = [f"{c[:25]}..." if len(c) > 25 else c for c in data_columns]
    header = "| Node | " + " | ".join(header_cols) + " | Final Score | Result |"
    separator = "|---| " + " | ".join(["---"] * len(header_cols)) + " |---|---|"
    
    lines.append(f"**Predicate:** `{predicate}`")
    lines.append(f"**Threshold:** `{threshold}` | **Top-K:** `{top_k}`")
    lines.append("")
    lines.append(header)
    lines.append(separator)
    
    # 4. detailed rows
    sorted_nodes = sorted(node_scores, key=lambda x: x.get("node_idx", 0))
    
    for node in sorted_nodes:
        name = node.get("node_name", "Unknown")
        final_score = node.get("final_score", 0.0)
        
        scores_to_display = []
        if node.get("scoring_steps"):
            scores_to_display = extract_atomic_scores(node["scoring_steps"][0])
        
        # Padding
        while len(scores_to_display) < len(data_columns):
            scores_to_display.append(0.0)
            
        scores_str = [f"{s:.4f}" for s in scores_to_display]
            
        final_score_str = f"{final_score:.4f}"
        
        # Result Status
        is_candidate = name in filtered_node_paths
        passed_threshold = final_score >= threshold
        
        status = "❌ Filtered Out"
        if is_candidate:
            status = "✅ Candidate"
        elif passed_threshold:
             status = "⚪ Above Threshold"
        else:
             # Try to explain generic failure
             if len(scores_to_display) == 1 and scores_to_display[0] > 0.8:
                 # Likely a NOT constraint match
                 status += " (Matches constraint)"

        row = f"| {name} | " + " | ".join(scores_str) + f" | {final_score_str} | {status} |"
        lines.append(row)

    lines.append("")
    return lines


def extract_semantic_xpath_data(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data.get("raw_result", {})
    
    # Extract XPath
    xpath = "N/A"
    parsed = raw.get("parsed_query", {})
    if "xpath" in parsed:
        xpath = parsed["xpath"]
    elif "xpath_execution" in raw and "query" in raw["xpath_execution"]:
        xpath = raw["xpath_execution"]["query"]

    # Extract Result Summary
    operation = data.get("operation", "UNKNOWN")
    summary = ""
    
    # Handler output (commonly used for reasoning/details)
    handler_output = raw.get("handler_result", {}).get("output", {})

    if operation == "READ":
        # READ results are typically promoted to top-level "selected_nodes"
        if "selected_nodes" in data and data["selected_nodes"]:
            count = len(data["selected_nodes"])
        elif "selected_nodes" in handler_output:
             count = len(handler_output["selected_nodes"])
        else:
            count = data.get("selected_count", 0)
        summary = f"Selected {count} nodes"
        
    elif operation == "DELETE":
        # DELETE count/paths often in raw_result or handler output
        if "deleted_count" in raw and raw["deleted_count"]:
            count = raw["deleted_count"]
        elif "deleted_paths" in raw and raw["deleted_paths"]:
            count = len(raw["deleted_paths"])
        elif "nodes_to_delete" in handler_output:
            count = len(handler_output["nodes_to_delete"])
        elif "deleted_paths" in handler_output:
             count = len(handler_output["deleted_paths"])
        else:
             count = data.get("deleted_count", 0)
        summary = f"Deleted {count} nodes"
        
    elif operation == "UPDATE":
        if "updated_count" in raw and raw["updated_count"]:
            count = raw["updated_count"]
        elif "updated_paths" in raw and raw["updated_paths"]:
            count = len(raw["updated_paths"])
        elif "updates" in handler_output:
            count = len(handler_output["updates"])
        else:
            count = data.get("updated_count", 0)
        summary = f"Updated {count} nodes"
        
    elif operation == "CREATE":
        path = raw.get("created_path")
        if not path:
             path = data.get("created_path")
        if not path and "parent_path" in handler_output and "node_type" in handler_output:
             path = f"{handler_output['parent_path']}/{handler_output['node_type']}"
        
        if not path:
             path = "unknown"
        summary = f"Created at {path}"
    else:
        summary = data.get("error", "Unknown result")

    # Extract Tokens
    token_usage = {}
    if "timing" in data and "total_tokens" in data["timing"]:
        token_usage = data["timing"]["total_tokens"]
    elif "token_usage" in data:
        token_usage = data["token_usage"]

    # Extract Time
    time_ms = 0
    if "timing" in data:
        time_ms = data["timing"].get("pipeline_total_ms", 0)
        if time_ms == 0:
            time_ms = data["timing"].get("total_time_ms", 0)
    else:
        time_ms = data.get("execution_time_ms", 0)

    return {
        "xpath": xpath,
        "result": summary,
        "token_usage": token_usage,
        "time_ms": time_ms,
        "operation": operation
    }

def extract_incontext_data(data: Dict[str, Any]) -> Dict[str, Any]:
    # XPath is N/A for incontext
    xpath = "N/A (Full Tree)"

    # Extract Result Summary
    operation = data.get("operation", "UNKNOWN")
    summary = ""
    
    if "diff" in data and data["diff"]:
        summary = data["diff"].get("summary", "")
    elif "error" in data:
        summary = f"Error: {data['error']}"
    else:
        summary = data.get("reasoning", "")

    # Extract Tokens
    token_usage = data.get("token_usage", {})

    # Extract Time
    time_ms = data.get("execution_time_ms", 0)

    return {
        "xpath": xpath,
        "result": summary,
        "token_usage": token_usage,
        "time_ms": time_ms,
        "operation": operation
    }

def generate_markdown_report(experiment_name: str, pipelines: List[str]):
    base_dir = Path(__file__).parent / "experiment_results" / experiment_name
    
    if not base_dir.exists():
        print(f"Experiment directory not found: {base_dir}")
        return

    output_lines = [
        f"# Experiment Report: {experiment_name}",
        "",
    ]

    # Find all unique query folders across pipelines to order them
    query_folders = set()
    for pipeline in pipelines:
        pipeline_dir = base_dir / pipeline
        if pipeline_dir.exists():
            # New structure: query_XXX/result.json
            folders = glob.glob(str(pipeline_dir / "query_*"))
            for folder in folders:
                if Path(folder).is_dir():
                    query_folders.add(Path(folder).name)
            # Also check for legacy flat structure: query_XXX_result.json
            files = glob.glob(str(pipeline_dir / "query_*_result.json"))
            for f in files:
                # Extract query number and convert to folder name format
                query_num = Path(f).name.split('_')[1]
                query_folders.add(f"query_{query_num}")
    
    sorted_folders = sorted(list(query_folders))

    # Generate separate summary table for each pipeline
    for pipeline in pipelines:
        output_lines.append(f"## Summary: {pipeline}")
        output_lines.append("")
        
        # Different columns based on pipeline type
        if pipeline == "semantic_xpath":
            output_lines.append("| Query | NL Request | Operation | XPath Query | Tokens | Time (s) |")
            output_lines.append("|---|---|---|---|---|---|")
        else:
            # incontext pipeline - no XPath column
            output_lines.append("| Query | NL Request | Operation | Tokens | Time (s) |")
            output_lines.append("|---|---|---|---|---|")
        
        for folder_name in sorted_folders:
            query_id = folder_name.split('_')[1]
            
            # Try new structure first: query_XXX/result.json
            file_path = base_dir / pipeline / folder_name / "result.json"
            if not file_path.exists():
                # Fall back to legacy structure: query_XXX_result.json
                file_path = base_dir / pipeline / f"{folder_name}_result.json"
            if not file_path.exists():
                continue
            
            data = load_result_json(file_path)
            
            if pipeline == "semantic_xpath":
                info = extract_semantic_xpath_data(data)
            elif pipeline == "incontext":
                info = extract_incontext_data(data)
            else:
                info = {
                    "xpath": "-", 
                    "result": "Unknown Pipeline", 
                    "token_usage": {}, 
                    "time_ms": 0, 
                    "operation": "?"
                }
            
            # NL Request - escape pipe characters and truncate if very long
            nl_request = data.get("query", "").replace("\n", " ").replace("|", "\\|")
            if len(nl_request) > 80:
                nl_request = nl_request[:77] + "..."
            
            token_str = format_token_usage(info['token_usage'])
            time_str = f"{info['time_ms'] / 1000:.2f}"
            
            if pipeline == "semantic_xpath":
                # Show full XPath query (escape pipes)
                xpath_query = info['xpath'].replace("|", "\\|") if info['xpath'] else "N/A"
                row = f"| {query_id} | {nl_request} | {info['operation']} | `{xpath_query}` | {token_str} | {time_str} |"
            else:
                # Incontext - no XPath column
                row = f"| {query_id} | {nl_request} | {info['operation']} | {token_str} | {time_str} |"
            
            output_lines.append(row)
        
        output_lines.append("")
        
        output_lines.append("")
        
        # Add stats table (pipeline specific)
        if pipeline == "semantic_xpath":
            stage_lines = format_stage_breakdown(base_dir, pipeline)
            output_lines.extend(stage_lines)
        elif pipeline == "incontext":
            stats_lines = format_incontext_stats(base_dir, pipeline)
            output_lines.extend(stats_lines)

    # Detailed sections (Optional)
    output_lines.append("")
    output_lines.append("## Detailed Results")
    
    for folder_name in sorted_folders:
        query_id = folder_name.split('_')[1]
        output_lines.append(f"### Query {query_id}")
        
        # Get query text from first available pipeline
        query_text = ""
        for pipeline in pipelines:
            # Try new structure first
            file_path = base_dir / pipeline / folder_name / "result.json"
            if not file_path.exists():
                # Fall back to legacy structure
                file_path = base_dir / pipeline / f"{folder_name}_result.json"
            if file_path.exists():
                data = load_result_json(file_path)
                query_text = data.get("query", "")
                break
        
        output_lines.append(f"**Query:** {query_text}")
        output_lines.append("")
        
        for pipeline in pipelines:
            # Try new structure first
            file_path = base_dir / pipeline / folder_name / "result.json"
            if not file_path.exists():
                # Fall back to legacy structure
                file_path = base_dir / pipeline / f"{folder_name}_result.json"
            if not file_path.exists():
                continue
                
            data = load_result_json(file_path)
            if pipeline == "semantic_xpath":
                info = extract_semantic_xpath_data(data)
            else:
                info = extract_incontext_data(data)

            output_lines.append(f"#### {pipeline}")
            output_lines.append(f"- **Operation:** {info['operation']}")
            # Only show XPath for semantic_xpath pipeline
            if pipeline == "semantic_xpath" and info['xpath']:
                output_lines.append(f"- **XPath:** `{info['xpath']}`")
            output_lines.append(f"- **Time:** {info['time_ms'] / 1000:.2f}s")
            output_lines.append(f"- **Tokens:** {format_token_usage(info['token_usage'])}")
            if "error" in data:
                output_lines.append(f"- **Error:** {data['error']}")
            output_lines.append("")

            # Display relevant nodes based on operation type
            operation = data.get("operation", "")
            
            if operation == "READ" and "selected_nodes" in data and data["selected_nodes"]:
                
                # Check if nodes need parsing (incontext pipeline returns raw XML)
                nodes_to_display = []
                for node in data["selected_nodes"]:
                    if "xml" in node and len(node) == 1:
                         # Parse the XML content
                         import xml.etree.ElementTree as ET
                         try:
                             # Wrap in wrapper to allow multiple roots
                             xml_content = node["xml"]
                             if xml_content.startswith("```"):
                                 xml_content = xml_content.split("\n", 1)[1].rsplit("\n", 1)[0]
                             
                             # Fix escaped quotes that may come from JSON serialization
                             xml_content = xml_content.replace('\\"', '"').replace("\\'", "'")
                             
                             root = ET.fromstring(f"<wrapper>{xml_content}</wrapper>")
                             
                             def generic_element_to_dict(elem):
                                 """Recursively convert XML element to dict."""
                                 data = {}
                                 # Attributes
                                 data.update(elem.attrib)
                                 
                                 # Children
                                 child_counts = {}
                                 for child in elem:
                                     child_counts[child.tag] = child_counts.get(child.tag, 0) + 1
                                 
                                 for child in elem:
                                     # Check if this child tag appears multiple times (structurally a list)
                                     is_list_item = child_counts[child.tag] > 1
                                     
                                     child_val = None
                                     if len(child) == 0:
                                         child_val = child.text.strip() if child.text else ""
                                     else:
                                         child_val = generic_element_to_dict(child)
                                     
                                     if is_list_item:
                                         if child.tag not in data:
                                             data[child.tag] = []
                                         if isinstance(data[child.tag], list):
                                             data[child.tag].append(child_val)
                                         else:
                                              data[child.tag] = [data[child.tag], child_val]
                                     else:
                                         data[child.tag] = child_val
                                 return data

                             for child in root:
                                 # Top level nodes (e.g. Day, or POI)
                                 node_data = generic_element_to_dict(child)
                                 node_data["type"] = child.tag
                                 
                                 # If it has sub-children that look like nodes, extract them for better display
                                 sub_nodes = []
                                 keys_to_remove = []
                                 
                                 # Keys to skip entirely (don't add as sub-nodes)
                                 skip_keys = {"highlights", "description", "type", "name", "indent"}
                                 
                                 for key, val in node_data.items():
                                     if key in skip_keys:
                                         continue
                                     if isinstance(val, list):
                                         # Check if list of dicts (complex objects)
                                         if val and isinstance(val[0], dict):
                                            # Treat as sub-nodes to display indented
                                            for item in val:
                                                item["type"] = key  # Use tag as type
                                                item["indent"] = True
                                                sub_nodes.append(item)
                                            keys_to_remove.append(key)
                                         elif not val:  # Empty list
                                            keys_to_remove.append(key)
                                     elif isinstance(val, dict):
                                         # complex child
                                         val["type"] = key
                                         val["indent"] = True
                                         sub_nodes.append(val)
                                         keys_to_remove.append(key)
                                 
                                 # Remove the moved keys so they don't duplicate
                                 for k in keys_to_remove:
                                     del node_data[k]
                                     
                                 nodes_to_display.append(node_data)
                                 nodes_to_display.extend(sub_nodes)

                         except Exception as e:
                             nodes_to_display.append({"name": "Error Parsing XML Node", "description": str(e)})
                    else:
                        nodes_to_display.append(node)

                output_lines.append("**Selected Nodes:**")
                output_lines.append("")
                
                # Different display format based on pipeline
                if pipeline == "semantic_xpath":
                    # Use a clean table format for semantic_xpath results
                    output_lines.append("| # | Node | Reasoning |")
                    output_lines.append("|---|------|-----------|")
                    
                    for idx, node in enumerate(nodes_to_display):
                        type_str = node.get('type', 'Node')
                        
                        # Build node identifier
                        if "index" in node:
                            node_id = f"{type_str} {node['index']}"
                        elif node.get('name') and node.get('name') != "Unknown":
                            node_id = f"{type_str}: {node['name']}"
                        elif "tree_path" in node:
                            # Extract last part of path
                            path_parts = node["tree_path"].split(" > ")
                            node_id = path_parts[-1] if path_parts else type_str
                        else:
                            node_id = type_str
                        
                        # Get reasoning, escape pipe characters
                        reasoning = node.get('reasoning', '-')
                        if reasoning:
                            reasoning = reasoning.replace("|", "\\|").replace("\n", " ")
                            if len(reasoning) > 150:
                                reasoning = reasoning[:147] + "..."
                        else:
                            reasoning = "-"
                        
                        output_lines.append(f"| {idx + 1} | {node_id} | {reasoning} |")
                    
                else:
                    # Incontext pipeline - hierarchical format
                    for node in nodes_to_display:
                        indent = node.get("indent", False)
                        prefix = "  - " if indent else "**"
                        suffix = "**" if not indent else ""
                        
                        type_str = node.get('type', 'Node')
                        name_str = node.get('name', '')
                        
                        # Build identity
                        if name_str and name_str != "Unknown":
                            identity = name_str
                        elif "index" in node:
                            identity = f"Index {node['index']}"
                        else:
                            identity = ""
                        
                        # Build compact details (filter empty values)
                        details = []
                        priority_keys = ["time_block", "expected_cost", "travel_method"]
                        for k in priority_keys:
                            val = node.get(k)
                            if val and str(val).strip():
                                details.append(str(val))
                        
                        details_str = f" ({', '.join(details)})" if details else ""
                        
                        if indent:
                            output_lines.append(f"  - {type_str}: {identity}{details_str}")
                        else:
                            node_display = f"{type_str} {identity}" if identity else type_str
                            output_lines.append(f"**{node_display}**{details_str}")
                
                output_lines.append("")
            
            elif operation == "DELETE" and "changes" in data and data["changes"]:
                output_lines.append("**Deleted Nodes:**")
                output_lines.append("")
                for change in data["changes"]:
                    if change.get("change_type") == "deleted":
                        path = change.get("path", "Unknown path")
                        output_lines.append(f"- `{path}`")
                output_lines.append("")
            
            elif operation == "CREATE" and "changes" in data and data["changes"]:
                output_lines.append("**Created Nodes:**")
                output_lines.append("")
                for change in data["changes"]:
                    if change.get("change_type") == "created" and "new_node" in change:
                        new_node = change["new_node"]
                        path = change.get("path", "")
                        output_lines.append(f"**Path:** `{path}`")
                        if "fields" in new_node:
                            output_lines.extend(format_node_details(new_node["fields"]))
                        output_lines.append("")

            # Special section for XPath execution details
            if pipeline == "semantic_xpath":
                # Look for execution trace file to get scoring details
                # It should be in query_XXX/reasoning_traces/execution_*.json
                
                # Check current directory structure first
                trace_dir = base_dir / pipeline / folder_name / "reasoning_traces"
                execution_trace = None
                
                if trace_dir.exists():
                    trace_files = glob.glob(str(trace_dir / "execution_*.json"))
                    if trace_files:
                        # Sort by name (timestamp) and take the last one
                        trace_files.sort()
                        execution_trace = load_result_json(Path(trace_files[-1]))
                
                if execution_trace:
                     scoring_table = format_scoring_table(execution_trace)
                     if scoring_table:
                         output_lines.append("**Scoring Analysis:**")
                         output_lines.append("")
                         output_lines.extend(scoring_table)
                if "xpath_execution" in data:
                    output_lines.append("<details>")
                    output_lines.append("<summary>Execution Details</summary>")
                    output_lines.append("")
                    output_lines.append(f"- Matched Nodes: {data['xpath_execution'].get('matched_count', 0)}")
                    output_lines.append(f"- Execution Time: {data['xpath_execution'].get('execution_time_ms', 0)}ms")
                    output_lines.append("</details>")
                output_lines.append("")

    report_path = base_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown report from experiment results.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment folder")
    parser.add_argument("--pipelines", type=str, nargs="+", default=["semantic_xpath", "incontext"], help="List of pipelines to include")
    
    args = parser.parse_args()
    
    generate_markdown_report(args.experiment_name, args.pipelines)
