"""
Configuration endpoints.

GET /api/config - Get current configuration
PUT /api/config - Update configuration
"""

from flask import Blueprint, request, jsonify, current_app
from api.serializers import serialize_config

bp = Blueprint("config", __name__)


@bp.route("", methods=["GET"])
def get_config():
    """
    Get current pipeline configuration.
    
    Returns:
        Current config values
    """
    pipeline = current_app.pipeline
    
    config = {
        "xpath_executor": {
            "top_k": pipeline.executor.executor.top_k,
            "score_threshold": pipeline.executor.executor.score_threshold,
            "scoring_method": pipeline.executor.executor.scoring_method
        },
        "mode": pipeline.mode,
        "active_schema": pipeline.executor.executor.schema_name,
        "active_data": pipeline.executor.executor.data_name or ""
    }
    
    return jsonify({
        "success": True,
        "config": serialize_config(config)
    })


@bp.route("", methods=["PUT"])
def update_config():
    """
    Update pipeline configuration.
    
    Request body:
        {
            "scoringMethod": "entailment" | "llm" | "cosine",
            "topK": 5,
            "scoreThreshold": 0.3
        }
        
    Returns:
        Updated config values
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    pipeline = current_app.pipeline
    executor = pipeline.executor.executor
    
    # Update values if provided
    if "scoringMethod" in data:
        method = data["scoringMethod"]
        if method not in ("llm", "entailment", "cosine"):
            return jsonify({"error": f"Invalid scoring method: {method}"}), 400
        # Note: Changing scoring method requires reinitializing the scorer
        # For now, we just note it would require restart
        
    if "topK" in data:
        try:
            executor.top_k = int(data["topK"])
        except (TypeError, ValueError):
            return jsonify({"error": "topK must be an integer"}), 400
            
    if "scoreThreshold" in data:
        try:
            executor.score_threshold = float(data["scoreThreshold"])
        except (TypeError, ValueError):
            return jsonify({"error": "scoreThreshold must be a number"}), 400
    
    # Return updated config
    config = {
        "xpath_executor": {
            "top_k": executor.top_k,
            "score_threshold": executor.score_threshold,
            "scoring_method": executor.scoring_method
        },
        "mode": pipeline.mode
    }
    
    return jsonify({
        "success": True,
        "message": "Configuration updated",
        "config": serialize_config(config)
    })
