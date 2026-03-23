"""
Schema Loader - Loads tree schema configurations.

Provides functions to load schema definitions and resolve data file paths.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


# Base directories
_BASE_DIR = Path(__file__).parent.parent.parent
_SCHEMA_DIR = _BASE_DIR / "storage" / "schemas"
_STORAGE_DIR = _BASE_DIR / "storage"


def _get_schema_dir(schema_name: str) -> Path:
    """
    Resolve the schema directory for a scenario.
    
    Supports both legacy single-file schemas and new folder-based schemas:
    - storage/schemas/{schema_name}.yaml
    - storage/schemas/{schema_name}/{schema_name}.yaml
    """
    candidate = _SCHEMA_DIR / schema_name
    if candidate.is_dir():
        return candidate
    return _SCHEMA_DIR


def _get_schema_path(schema_name: str) -> Path:
    """Get path to the main content schema file."""
    return _get_schema_dir(schema_name) / f"{schema_name}.yaml"


def _get_version_schema_path(schema_name: str) -> Path:
    """Get path to the version schema file for a scenario."""
    return _get_schema_dir(schema_name) / f"{schema_name}_version.yaml"


def load_config() -> Dict[str, Any]:
    """Load the main config.yaml file."""
    config_path = _BASE_DIR / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_schema(schema_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a schema definition by name.
    
    Args:
        schema_name: Name of the schema (e.g., "itinerary"). 
                    If None, uses active_schema from config.
        config: Optional config dict. If not provided, loads from config.yaml.
    
    Returns:
        Schema dictionary with node definitions, data files, etc.
    """
    if schema_name is None:
        if config is None:
            config = load_config()
        schema_name = config.get("active_schema")
        if not schema_name:
            raise ValueError("No active_schema set in config")
    
    schema_path = _get_schema_path(schema_name)
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)


def load_version_schema(schema_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load the version schema definition by name.
    
    Args:
        schema_name: Name of the schema (e.g., "itinerary").
                    If None, uses active_schema from config.
        config: Optional config dict. Used to resolve schema_name if not provided.
    
    Returns:
        Version schema dictionary, or empty dict if not found.
    """
    if schema_name is None:
        if config is None:
            config = load_config()
        schema_name = config.get("active_schema")
        if not schema_name:
            raise ValueError("No active_schema set in config")
    
    version_path = _get_version_schema_path(schema_name)
    
    if not version_path.exists():
        return {}
    
    with open(version_path, "r") as f:
        return yaml.safe_load(f)


def _find_node_by_type(nodes: Dict[str, Any], node_type: str) -> Optional[str]:
    for name, config in nodes.items():
        if config.get("type") == node_type:
            return name
    return None


def _find_path_between_nodes(
    nodes: Dict[str, Any], 
    root_name: str, 
    target_name: str
) -> list:
    """Find a path from root to target using children links."""
    if not root_name or not target_name:
        return []
    
    queue = [(root_name, [root_name])]
    visited = set()
    
    while queue:
        current, path = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        if current == target_name:
            return path
        
        children = nodes.get(current, {}).get("children", [])
        for child in children:
            if child not in visited:
                queue.append((child, path + [child]))
    
    return []


def get_versioning_info(schema_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get versioning metadata and resolved path info for a schema.
    
    Args:
        schema_name: Name of the schema. If None, resolved from config.
        config: Optional config dict. Used to resolve schema_name if not provided.
    
    Returns:
        Dict with root tag, version tag, index attr, and path parts.
    """
    version_schema = load_version_schema(schema_name, config=config)
    nodes = version_schema.get("nodes", {}) if version_schema else {}
    versioning = version_schema.get("versioning", {}) if version_schema else {}
    
    root_tag = versioning.get("root") or _find_node_by_type(nodes, "root")
    version_tag = versioning.get("version_node") or _find_node_by_type(nodes, "version")
    
    index_attr = (
        versioning.get("version_index_attr")
        or nodes.get(version_tag, {}).get("index_attr")
        or "number"
    )
    
    content_container = versioning.get("content_container")
    
    path_parts = versioning.get("path_parts")
    if not path_parts and root_tag and version_tag:
        path_parts = _find_path_between_nodes(nodes, root_tag, version_tag)
    if not path_parts and root_tag and version_tag:
        path_parts = [root_tag, version_tag]
    
    version_path = "/" + "/".join(path_parts) if path_parts else ""
    
    return {
        "root_tag": root_tag,
        "version_tag": version_tag,
        "version_index_attr": index_attr,
        "content_container": content_container,
        "version_path_parts": path_parts,
        "version_path": version_path,
    }


def get_data_path(
    data_name: Optional[str] = None, 
    schema_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Get the full path to a data file.
    
    Args:
        data_name: Name of the data file (e.g., "travel_memory_5day").
                  If None, uses active_data from config or schema's default.
        schema_name: Name of the schema. If None, uses active_schema from config.
        config: Optional config dict. If not provided, loads from config.yaml.
    
    Returns:
        Full Path to the data file.
    """
    if schema_name is None and config is None:
        config = load_config()
    if schema_name is None and config is not None:
        schema_name = config.get("active_schema")
    schema = load_schema(schema_name, config=config)
    
    # Determine which data file to use
    if data_name is None:
        # First check config for active_data
        if config is not None:
            data_name = config.get("active_data")
        
        # Fall back to schema's default_data
        if data_name is None:
            data_name = schema.get("default_data")
    
    if data_name is None:
        raise ValueError("No data file specified and no default found")
    
    # Get the relative path from schema
    data_files = schema.get("data_files", {})
    
    if data_name not in data_files:
        raise ValueError(
            f"Data file '{data_name}' not found in schema. "
            f"Available: {list(data_files.keys())}"
        )
    
    relative_path = data_files[data_name]
    return _STORAGE_DIR / relative_path


def get_schema_info(schema_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get comprehensive schema information including resolved paths.
    
    Args:
        schema_name: Name of the schema. If None, uses active_schema from config.
        config: Optional config dict. If not provided and schema_name is None,
                falls back to config.yaml.
    
    Returns:
        Dictionary with schema info and resolved paths.
    """
    if schema_name is None:
        if config is None:
            config = load_config()
        schema_name = config.get("active_schema")
        if not schema_name:
            raise ValueError("No active_schema found in config")
    
    schema = load_schema(schema_name)
    version_schema = load_version_schema(schema_name)
    versioning_info = get_versioning_info(schema_name)
    
    # Get active data name from config if available, else from schema default
    active_data = None
    if config:
        active_data = config.get("active_data")
    if not active_data:
        active_data = schema.get("default_data")
    
    # Resolve all data file paths
    resolved_data_files = {}
    for name, rel_path in schema.get("data_files", {}).items():
        resolved_data_files[name] = str(_STORAGE_DIR / rel_path)
    
    # Determine content root for prompt guidance
    content_root = _find_node_by_type(schema.get("nodes", {}), "root")
    
    return {
        "schema_name": schema.get("name"),
        "description": schema.get("description"),
        "hierarchy": schema.get("hierarchy"),
        "nodes": schema.get("nodes", {}),
        "active_data": active_data,
        "data_files": resolved_data_files,
        "examples_file": str(_STORAGE_DIR / schema.get("examples_file", "")),
        "syntax_rules": schema.get("syntax_rules", ""),
        "version_resolver_prompt": schema.get(
            "version_resolver_prompt",
            "prompts/query_generator/version_resolver.txt",
        ),
        "version_schema": version_schema,
        "versioning": versioning_info,
        "content_root": content_root,
    }


def get_node_config(schema_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get node configuration for NodeUtils.
    
    Args:
        schema_name: Name of the schema. If None, uses active_schema from config.
    
    Returns:
        Dictionary mapping node types to their configuration.
    """
    schema = load_schema(schema_name)
    return schema.get("nodes", {})


def list_available_schemas() -> list:
    """List all available schema names."""
    schemas = set()
    
    # Legacy single-file schemas
    for path in _SCHEMA_DIR.glob("*.yaml"):
        schemas.add(path.stem)
    
    # Folder-based schemas
    for item in _SCHEMA_DIR.iterdir():
        if item.is_dir():
            content_path = item / f"{item.name}.yaml"
            if content_path.exists():
                schemas.add(item.name)
    
    return sorted(schemas)


def list_available_data_files(schema_name: Optional[str] = None) -> list:
    """List available data files for a schema."""
    schema = load_schema(schema_name)
    return list(schema.get("data_files", {}).keys())


def get_schema_summary_for_prompt(schema_name: Optional[str] = None) -> str:
    """
    Generate a schema summary for LLM prompts.
    
    Shows the tree hierarchy with node types and their available fields.
    This helps the LLM understand what information is available at each node
    and determine when to use child vs desc axis.
    
    Args:
        schema_name: Name of the schema. If None, uses active_schema from config.
    
    Returns:
        Formatted string describing the schema structure and fields.
    """
    schema = load_schema(schema_name)
    nodes = schema.get("nodes", {})
    hierarchy = schema.get("hierarchy", "")
    
    lines = []
    
    # Include hierarchy visualization if available (helps with axis selection)
    if hierarchy:
        lines.append("Tree Hierarchy:")
        lines.append(hierarchy.strip())
        lines.append("")
    
    lines.append("Node Definitions:")
    lines.append("")
    
    for node_name, node_config in nodes.items():
        node_type = node_config.get("type", "unknown")
        fields = node_config.get("fields", [])
        children = node_config.get("children", [])
        index_attr = node_config.get("index_attr")
        
        # Node header
        if node_type == "root":
            lines.append(f"{node_name} (root)")
        elif node_type == "container":
            if index_attr:
                lines.append(f"{node_name} (container, indexed by @{index_attr})")
            else:
                lines.append(f"{node_name} (container)")
        else:
            lines.append(f"{node_name} (leaf)")
        
        # Fields
        if fields:
            lines.append(f"  Fields: {', '.join(fields)}")
        
        # Children - crucial for axis selection
        if children:
            lines.append(f"  Direct children: {', '.join(children)}")
        
        lines.append("")
    
    return "\n".join(lines)
