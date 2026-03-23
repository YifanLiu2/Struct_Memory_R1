"""
Flask application factory for the Semantic XPath API.

Creates and configures the Flask app with:
- CORS support for React frontend
- Blueprint registration
- Pipeline initialization
"""

from flask import Flask
from flask_cors import CORS
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_app(config: dict = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    
    # Default configuration
    app.config.update({
        "SCORING_METHOD": "entailment",
        "TOP_K": 5,
        "SCORE_THRESHOLD": 0.3,
    })
    
    # Apply overrides
    if config:
        app.config.update(config)
    
    # Enable CORS for React frontend
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Initialize pipeline as app extension
    from pipeline import SemanticXPathPipeline
    
    app.pipeline = SemanticXPathPipeline(
        scoring_method=app.config.get("SCORING_METHOD"),
        top_k=app.config.get("TOP_K"),
        score_threshold=app.config.get("SCORE_THRESHOLD")
    )
    
    # Register blueprints
    from .routes import query_bp, tree_bp, config_bp
    
    app.register_blueprint(query_bp, url_prefix="/api")
    app.register_blueprint(tree_bp, url_prefix="/api/tree")
    app.register_blueprint(config_bp, url_prefix="/api/config")
    
    # Health check endpoint
    @app.route("/api/health")
    def health():
        return {"status": "ok", "mode": "demo"}
    
    return app
