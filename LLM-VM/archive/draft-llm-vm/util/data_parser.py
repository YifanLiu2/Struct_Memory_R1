"""
Data Parser utility for handling LLM responses and entity storage.
"""
import json
import re
from pathlib import Path
from typing import Optional


STORAGE_DIR = Path(__file__).parent.parent / "storage"


def load_entities() -> dict:
    """Load entities from storage."""
    entities_path = STORAGE_DIR / "entities.json"
    with open(entities_path, "r") as f:
        return json.load(f)


def load_manifest() -> str:
    """Load the registry manifest from storage."""
    manifest_path = STORAGE_DIR / "registry_nl_manifest.txt"
    with open(manifest_path, "r") as f:
        return f.read()


def clean_json_from_markdown(text: str) -> str:
    """
    Remove markdown code block wrappers from JSON text.
    
    Handles formats like:
    - ```json ... ```
    - ``` ... ```
    - ```python ... ```
    
    Args:
        text: Raw text that may contain markdown code blocks
        
    Returns:
        Cleaned text with code block markers removed
    """
    cleaned = text.strip()
    
    # Handle markdown code blocks with optional language specifier
    if cleaned.startswith("```"):
        # Remove opening ``` with optional language tag
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        # Remove closing ```
        cleaned = re.sub(r'\n?```$', '', cleaned)
    
    return cleaned.strip()


def parse_json_response(
    response: str,
    default: Optional[dict] = None):
    """
    Parse JSON from an LLM response, handling markdown wrappers.
    
    Args:
        response: Raw LLM response text
        default: Default value to return on parse failure (default: empty dict)
        
    Returns:
        Tuple of (parsed_result, error_message)
        - On success: (parsed_json, None)
        - On failure: (default_value, error_message)
    """
    if default is None:
        default = {}
    
    try:
        cleaned = clean_json_from_markdown(response)
        result = json.loads(cleaned)
        return result, None
    except json.JSONDecodeError as e:
        return default, f"JSON parse error: {str(e)}"


def _lookup_entity(entity_id: str, entities: dict) -> Optional[dict]:
    """
    Look up entity details by ID.
    
    Args:
        entity_id: Entity ID (e.g., "poi_001", "rest_001")
        entities: Entity dictionary with categories
        
    Returns:
        Entity details dict or None if not found
    """
    if entity_id.startswith("poi_"):
        return entities.get("poi", {}).get(entity_id)
    elif entity_id.startswith("rest_"):
        return entities.get("restaurant", {}).get(entity_id)
    return None


def resolve_pointers_in_json(
    data: dict,
    pointer_field: str = "entity_id"
):
    """
    Resolve entity pointers within a JSON structure.
    
    Traverses the JSON and replaces entity references with full entity data.
    
    Args:
        data: JSON structure containing entity pointers
        pointer_field: Field name that contains entity ID references
        
    Returns:
        JSON structure with resolved entity details
    """
    entities = load_entities()
    
    if isinstance(data, dict):
        resolved = {}
        for key, value in data.items():
            if key == pointer_field and isinstance(value, str):
                # Resolve the entity pointer
                entity_details = _lookup_entity(value, entities)
                if entity_details:
                    resolved[key] = value
                    resolved["entity_details"] = entity_details
                else:
                    resolved[key] = value
            else:
                resolved[key] = resolve_pointers_in_json(value, pointer_field)
        return resolved
    
    elif isinstance(data, list):
        return [resolve_pointers_in_json(item, pointer_field) for item in data]
    
    else:
        return data


def _collect_entity_ids(
    data: dict,
    id_fields: tuple[str, ...],
    collected: set
) -> None:
    """
    Recursively collect entity IDs from a JSON structure.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in id_fields:
                if isinstance(value, str):
                    collected.add(value)
                elif isinstance(value, list):
                    collected.update(v for v in value if isinstance(v, str))
            else:
                _collect_entity_ids(value, id_fields, collected)
    
    elif isinstance(data, list):
        for item in data:
            _collect_entity_ids(item, id_fields, collected)


def get_related_entities(
    data: dict,
    id_fields: tuple[str, ...] = ("matched_ids", "entity_id", "entity_ids")
) -> dict:
    """
    Extract and resolve all entity references from parsed JSON data.
    
    Args:
        data: Parsed JSON response data
        id_fields: Field names that may contain entity ID(s)
        
    Returns:
        Dictionary mapping entity IDs to their full details
    """
    entities = load_entities()
    entity_ids = set()
    _collect_entity_ids(data, id_fields, entity_ids)
    
    details = {}
    for eid in entity_ids:
        entity_data = _lookup_entity(eid, entities)
        if entity_data:
            details[eid] = entity_data
    
    return details


def get_entity_details(entity_ids: list) -> dict:
    """
    Retrieve full entity details for given IDs.
    
    Args:
        entity_ids: List of entity IDs to look up
        
    Returns:
        Dictionary mapping entity IDs to their full details
    """
    entities = load_entities()
    details = {}
    
    for eid in entity_ids:
        entity_data = _lookup_entity(eid, entities)
        if entity_data:
            details[eid] = entity_data
    
    return details


def get_entities_json() -> str:
    """
    Get entities as a formatted JSON string for LLM prompts.
    
    Returns:
        JSON string of all entities
    """
    entities = load_entities()
    return json.dumps(entities, indent=2)
