"""
Flask API for the Semantic XPath Pipeline.

Provides REST endpoints for:
- Executing CRUD queries
- Managing tree state
- Configuring pipeline settings

Run with: python -m api.run
"""

from .app import create_app

__all__ = ["create_app"]
