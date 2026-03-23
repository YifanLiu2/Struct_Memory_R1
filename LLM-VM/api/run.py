"""
Flask server entry point.

Run with: python -m api.run [options]

Options:
    --port PORT         Port to run on (default: 5000)
    --host HOST         Host to bind to (default: 127.0.0.1)
    --scoring METHOD    Scoring method: llm, entailment, cosine (default: entailment)
    --top-k K           Number of top results (default: 5)
    --threshold T       Score threshold (default: 0.3)
    --debug             Enable debug mode
"""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Semantic XPath Pipeline API Server"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to run on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--scoring", "-s",
        choices=["llm", "entailment", "cosine"],
        default="entailment",
        help="Scoring method (default: entailment)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Score threshold (default: 0.3)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create app with config
    config = {
        "SCORING_METHOD": args.scoring,
        "TOP_K": args.top_k,
        "SCORE_THRESHOLD": args.threshold
    }
    
    app = create_app(config)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          Semantic XPath Pipeline API Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  Mode:      DEMO (tree persists across queries)              ║
║  Scoring:   {args.scoring:<47} ║
║  Top-K:     {args.top_k:<47} ║
║  Threshold: {args.threshold:<47} ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /api/query        Execute a CRUD query               ║
║    GET  /api/tree         Get current tree state             ║
║    POST /api/tree/reset   Reset tree to original             ║
║    GET  /api/config       Get configuration                  ║
║    PUT  /api/config       Update configuration               ║
╠══════════════════════════════════════════════════════════════╣
║  Server: http://{args.host}:{args.port:<38} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
