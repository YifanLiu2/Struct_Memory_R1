"""
Tree state endpoints.

GET /api/tree - Get current tree state
GET /api/tree/versions - List all tree versions  
GET /api/tree/version/<id> - Get specific version
POST /api/tree/reset - Reset tree to original state
"""

from flask import Blueprint, jsonify, current_app
from pathlib import Path

from api.serializers import serialize_tree

bp = Blueprint("tree", __name__)


def get_executor(pipeline):
    """Get the executor from the pipeline (handles both direct and orchestrator access)."""
    if hasattr(pipeline, 'orchestrator') and hasattr(pipeline.orchestrator, 'executor'):
        return pipeline.orchestrator.executor
    elif hasattr(pipeline, 'executor'):
        return pipeline.executor
    else:
        raise AttributeError("Cannot find executor in pipeline")


@bp.route("", methods=["GET"])
def get_tree():
    """
    Get the current tree state as JSON.
    
    Returns:
        Serialized tree with node structure
    """
    pipeline = current_app.pipeline
    
    try:
        executor = get_executor(pipeline)
        tree_data = serialize_tree(executor.tree.getroot())
        return jsonify({
            "success": True,
            "tree": tree_data,
            "dataFile": executor.memory_path.name if hasattr(executor, 'memory_path') else "unknown"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@bp.route("/versions", methods=["GET"])
def list_versions():
    """
    List all saved tree versions.
    
    Returns:
        List of version metadata
    """
    pipeline = current_app.pipeline
    
    try:
        executor = get_executor(pipeline)
        # Get versions from the version manager if available
        if hasattr(executor, 'version_manager') and hasattr(executor, 'memory_path'):
            versions = executor.version_manager.list_versions(executor.memory_path)
        else:
            versions = []
        
        return jsonify({
            "success": True,
            "versions": versions
        })
    except Exception as e:
        # If no versions exist yet, return empty list
        return jsonify({
            "success": True,
            "versions": []
        })


@bp.route("/version/<int:version_id>", methods=["GET"])
def get_version(version_id: int):
    """
    Get a specific tree version.
    
    Args:
        version_id: Version number to retrieve
        
    Returns:
        Serialized tree at that version
    """
    pipeline = current_app.pipeline
    
    try:
        executor = get_executor(pipeline)
        # Build version file path
        if hasattr(executor, "memory_path"):
            base_name = executor.memory_path.stem
        else:
            base_name = getattr(executor, "schema_name", None) or "tree"
        result_dir = Path(__file__).parent.parent.parent / "result" / "demo"
        version_file = result_dir / f"{base_name}_v{version_id}.xml"
        
        if not version_file.exists():
            return jsonify({
                "error": f"Version {version_id} not found",
                "success": False
            }), 404
        
        import xml.etree.ElementTree as ET
        tree = ET.parse(version_file)
        tree_data = serialize_tree(tree.getroot())
        
        return jsonify({
            "success": True,
            "version": version_id,
            "tree": tree_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@bp.route("/reset", methods=["POST"])
def reset_tree():
    """
    Reset the tree to its original state.
    
    Returns:
        Fresh tree state
    """
    pipeline = current_app.pipeline
    
    try:
        # Try orchestrator method first, then executor method
        if hasattr(pipeline, 'orchestrator') and hasattr(pipeline.orchestrator, 'reload_tree'):
            pipeline.orchestrator.reload_tree()
        elif hasattr(pipeline, 'executor') and hasattr(pipeline.executor, 'reload_tree'):
            pipeline.executor.reload_tree()
        else:
            return jsonify({"error": "No reload method available", "success": False}), 500
        
        # Get fresh tree state
        executor = get_executor(pipeline)
        tree_data = serialize_tree(executor.tree.getroot())
        
        return jsonify({
            "success": True,
            "message": "Tree reset to original state",
            "tree": tree_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500
