"""
Query endpoint for executing CRUD operations.

POST /api/query - Execute a natural language query
"""

from flask import Blueprint, request, jsonify, current_app
from api.serializers import serialize_result, serialize_tree

bp = Blueprint("query", __name__)


def get_executor(pipeline):
    """Get the executor from the pipeline (handles both direct and orchestrator access)."""
    if hasattr(pipeline, 'orchestrator') and hasattr(pipeline.orchestrator, 'executor'):
        return pipeline.orchestrator.executor
    elif hasattr(pipeline, 'executor'):
        return pipeline.executor
    else:
        raise AttributeError("Cannot find executor in pipeline")


@bp.route("/query", methods=["POST"])
def execute_query():
    """
    Execute a natural language CRUD query.
    
    Request body:
        {
            "query": "find museums in the itinerary"
        }
        
    Returns:
        Full pipeline result with tree state before/after, including:
        - traversalSteps: Step-by-step query execution with scoring details
        - scoreFusion: How scores from multiple predicates combine
        - finalFiltering: TopK and threshold filtering details
    """
    data = request.get_json()
    
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    pipeline = current_app.pipeline
    
    try:
        executor = get_executor(pipeline)
        
        # Capture tree state before operation
        tree_before = serialize_tree(executor.tree.getroot())
        
        # Execute the query
        result = pipeline.process_request(query)
        
        # Capture tree state after operation
        tree_after = serialize_tree(executor.tree.getroot())
        
        # Build response with all visualization data
        response = serialize_result(result)
        response["tree"] = {
            "before": tree_before,
            "after": tree_after
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
