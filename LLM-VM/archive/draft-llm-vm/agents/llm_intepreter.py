from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.openai_client import make_request
from util.data_parser import (
    parse_json_response,
    get_entity_details,
    load_manifest,
)


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
REASONING_TRACES_DIR = Path(__file__).parent.parent / "reasoning_traces"


def load_prompt(name: str) -> str:
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    with open(prompt_path, "r") as f:
        return f.read()


def _generate_trace_id(user_request: str) -> str:
    """Generate a unique folder name based on request and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{user_request}"


def _save_trace(trace_folder: Path, filename: str, data: dict) -> None:
    """Save trace data to a JSON file in the trace folder."""
    trace_folder.mkdir(parents=True, exist_ok=True)
    file_path = trace_folder / f"{filename}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def _run_stage1(user_request: str, manifest: str) -> dict:
    """
    Stage 1: Scan the manifest to identify candidate entity IDs.
    
    Returns dict with prompts, response, and parsed result.
    """
    system_prompt = load_prompt("stage1_manifest_system")
    user_prompt_template = load_prompt("stage1_manifest_user")
    
    user_prompt = user_prompt_template.format(
        user_request=user_request,
        manifest=manifest
    )
    
    response = make_request(user_prompt, system_prompt)
    
    default_result = {
        "candidate_ids": [],
        "stage1_reasoning": "Failed to parse Stage 1 response"
    }
    
    parsed, error = parse_json_response(response, default=default_result)
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": response,
        "parsed_result": parsed,
        "parse_error": error
    }


def _run_stage2(
    user_request: str,
    manifest: str,
    candidate_ids: list,
    stage1_reasoning: str,
    candidate_entities: dict
) -> dict:
    """
    Stage 2: Resolve which candidates are truly relevant using full entity details.
    
    Returns dict with prompts, response, and parsed result.
    """
    system_prompt = load_prompt("stage2_resolve_system")
    user_prompt_template = load_prompt("stage2_resolve_user")
    
    user_prompt = user_prompt_template.format(
        user_request=user_request,
        manifest=manifest,
        candidate_ids=json.dumps(candidate_ids, indent=2),
        stage1_reasoning=stage1_reasoning,
        candidate_entities=json.dumps(candidate_entities, indent=2)
    )
    
    response = make_request(user_prompt, system_prompt)
    
    default_result = {
        "matched_ids": [],
        "stage2_reasoning": "Failed to parse Stage 2 response",
        "filtered_out": {},
        "confidence": "low"
    }
    
    parsed, error = parse_json_response(response, default=default_result)
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": response,
        "parsed_result": parsed,
        "parse_error": error
    }


def resolve_pointer(user_request: str) -> dict:
    """
    Use LLM to determine which POI(s) or restaurant(s) are related to the user request.
    
    Works in two stages:
    1. Stage 1: Quick manifest scan to identify candidate IDs
    2. Stage 2: Detailed entity review to confirm truly relevant items
    
    Saves full prompts and results to reasoning_traces folder.
    
    Args:
        user_request: Natural language request from user
        
    Returns:
        dict with matched entity IDs, related entities, and trace folder location
    """
    # Create trace folder for this request
    trace_id = _generate_trace_id(user_request)
    trace_folder = REASONING_TRACES_DIR / trace_id
    
    # Load manifest once
    manifest = load_manifest()
    
    # ==================== STAGE 1 ====================
    stage1 = _run_stage1(user_request, manifest)
    
    candidate_ids = stage1["parsed_result"].get("candidate_ids", [])
    stage1_reasoning = stage1["parsed_result"].get("stage1_reasoning", "")
    
    # Save Stage 1 trace
    _save_trace(trace_folder, "stage1", {
        "system_prompt": stage1["system_prompt"],
        "user_prompt": stage1["user_prompt"],
        "raw_response": stage1["raw_response"],
        "parsed_result": stage1["parsed_result"],
        "parse_error": stage1["parse_error"]
    })
    
    # ==================== STAGE 2 ====================
    candidate_entities = get_entity_details(candidate_ids)
    
    stage2 = _run_stage2(
        user_request=user_request,
        manifest=manifest,
        candidate_ids=candidate_ids,
        stage1_reasoning=stage1_reasoning,
        candidate_entities=candidate_entities
    )
    
    matched_ids = stage2["parsed_result"].get("matched_ids", [])
    matched_entities = get_entity_details(matched_ids)
    
    # Save Stage 2 trace
    _save_trace(trace_folder, "stage2", {
        "system_prompt": stage2["system_prompt"],
        "user_prompt": stage2["user_prompt"],
        "raw_response": stage2["raw_response"],
        "parsed_result": stage2["parsed_result"],
        "parse_error": stage2["parse_error"]
    })
    
    # ==================== FINAL RESULT ====================
    final_result = {
        "user_request": user_request,
        "matched_ids": matched_ids,
        "matched_entities": matched_entities,
        "confidence": stage2["parsed_result"].get("confidence", "low"),
        "reasoning_trace": {
            "stage1": {
                "candidate_ids": candidate_ids,
                "reasoning": stage1_reasoning
            },
            "stage2": {
                "reasoning": stage2["parsed_result"].get("stage2_reasoning", ""),
                "filtered_out": stage2["parsed_result"].get("filtered_out", {})
            }
        },
        "trace_folder": str(trace_folder)
    }
    
    # Save final summary
    _save_trace(trace_folder, "summary", final_result)
    
    return final_result
