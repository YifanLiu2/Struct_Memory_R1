"""
Session Manager - Manages CLI session-based logging structure.

Creates and manages the folder structure:
cli_session_results/
  └── Session001/
      ├── tree.xml (copied tree for modification)
      └── query001/
          └── query-related logs
      └── query002/
          └── ...
  └── Session002/
      └── ...
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """
    Find project root by looking for config.yaml.
    
    Returns:
        Path to project root directory
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config.yaml").exists():
            return parent
    raise RuntimeError("Could not find project root (config.yaml not found)")


class SessionManager:
    """
    Manages CLI session-based logging structure.
    
    Creates session folders with:
    - Copied tree for modification
    - Query-specific log folders
    - Session summary information
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the session manager.
        
        Args:
            base_dir: Base directory for session results. 
                      Defaults to cli_session_results in project root.
        """
        if base_dir is None:
            project_root = _find_project_root()
            base_dir = project_root / "cli_session_results"
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session_dir: Optional[Path] = None
        self.current_session_id: Optional[str] = None
        self.current_query_count: int = 0
        self.session_start_time: Optional[datetime] = None
    
    def start_session(self) -> Path:
        """
        Start a new session and create the session folder.
        
        Returns:
            Path to the session directory
        """
        session_id = self._get_next_session_id()
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session_dir = session_dir
        self.current_session_id = session_id
        self.current_query_count = 0
        self.session_start_time = datetime.now()
        
        logger.info(f"Started session: {session_id} at {session_dir}")
        return session_dir
    
    def copy_tree_to_session(self, source_tree_path: Path) -> Path:
        """
        Copy the source tree to the session folder.
        
        Args:
            source_tree_path: Path to the source XML tree file
            
        Returns:
            Path to the copied tree file in the session folder
        """
        if self.current_session_dir is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        dest_tree_path = self.current_session_dir / "tree.xml"
        shutil.copy2(source_tree_path, dest_tree_path)
        
        logger.info(f"Copied tree to session: {dest_tree_path}")
        return dest_tree_path
    
    def start_query(self) -> Path:
        """
        Start a new query within the current session.
        
        Creates a query folder and returns the path.
        
        Returns:
            Path to the query directory
        """
        if self.current_session_dir is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        self.current_query_count += 1
        query_id = f"query{self.current_query_count:03d}"
        query_dir = self.current_session_dir / query_id
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # Create reasoning_traces subfolder for trace files
        traces_dir = query_dir / "reasoning_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Started query: {query_id} at {query_dir}")
        return query_dir
    
    def get_current_query_dir(self) -> Optional[Path]:
        """
        Get the current query directory.
        
        Returns:
            Path to current query directory, or None if no active query
        """
        if self.current_session_dir is None or self.current_query_count == 0:
            return None
        
        query_id = f"query{self.current_query_count:03d}"
        return self.current_session_dir / query_id
    
    def save_query_log(
        self, 
        user_query: str, 
        result: Dict[str, Any],
        query_dir: Optional[Path] = None
    ):
        """
        Save the query log to the query folder.
        
        Args:
            user_query: The user's natural language query
            result: The pipeline result dictionary
            query_dir: Optional query directory (uses current if not provided)
        """
        if query_dir is None:
            query_dir = self.get_current_query_dir()
        
        if query_dir is None:
            logger.warning("No query directory available for saving logs")
            return
        
        # Save query master log
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "operation": result.get("operation"),
            "success": result.get("success"),
            "xpath_query": result.get("xpath_query"),
            "full_query": result.get("full_query"),
            "canonical_xpath_query": result.get("canonical_xpath_query"),
            "canonical_full_query": result.get("canonical_full_query"),
            "timing": result.get("timing", {}),
            "total_time_ms": result.get("total_time_ms"),
            "result_summary": self._extract_result_summary(result)
        }
        
        log_file = query_dir / "query_master_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved query log to {log_file}")
    
    def save_tree_snapshot(self, tree_content: str, query_dir: Optional[Path] = None):
        """
        Save a snapshot of the tree after a query.
        
        Args:
            tree_content: XML content of the tree
            query_dir: Optional query directory (uses current if not provided)
        """
        if query_dir is None:
            query_dir = self.get_current_query_dir()
        
        if query_dir is None:
            logger.warning("No query directory available for saving tree snapshot")
            return
        
        snapshot_file = query_dir / "tree_snapshot.xml"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            f.write(tree_content)
        
        logger.debug(f"Saved tree snapshot to {snapshot_file}")
    
    def end_session(self, stats: Optional[Dict[str, Any]] = None):
        """
        End the current session and save session summary.
        
        Args:
            stats: Optional session statistics dictionary
        """
        if self.current_session_dir is None:
            return
        
        # Calculate session duration
        session_end_time = datetime.now()
        duration_seconds = None
        if self.session_start_time:
            duration_seconds = (session_end_time - self.session_start_time).total_seconds()
        
        # Save session summary
        summary = {
            "session_id": self.current_session_id,
            "start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "end_time": session_end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "total_queries": self.current_query_count,
            "statistics": stats
        }
        
        summary_file = self.current_session_dir / "session_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session ended: {self.current_session_id}, {self.current_query_count} queries")
        
        # Reset state
        self.current_session_dir = None
        self.current_session_id = None
        self.current_query_count = 0
        self.session_start_time = None
    
    def _get_next_session_id(self) -> str:
        """
        Get the next available session ID.
        
        Returns:
            Session ID string (e.g., "Session001")
        """
        existing_sessions = sorted([
            d.name for d in self.base_dir.iterdir() 
            if d.is_dir() and d.name.startswith("Session")
        ])
        
        if not existing_sessions:
            return "Session001"
        
        # Extract the highest session number
        last_session = existing_sessions[-1]
        try:
            last_num = int(last_session.replace("Session", ""))
            return f"Session{last_num + 1:03d}"
        except ValueError:
            return f"Session{len(existing_sessions) + 1:03d}"
    
    def _extract_result_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a summary of the operation result.
        
        Args:
            result: Pipeline result dictionary
            
        Returns:
            Summary dictionary
        """
        operation = result.get("operation", "UNKNOWN")
        summary = {
            "operation": operation,
            "success": result.get("success", False)
        }
        
        if operation == "READ":
            summary["candidates_count"] = result.get("candidates_count", 0)
            summary["selected_count"] = result.get("selected_count", 0)
        elif operation == "DELETE":
            summary["deleted_count"] = result.get("deleted_count", 0)
            summary["deleted_paths"] = result.get("deleted_paths", [])
        elif operation == "UPDATE":
            summary["updated_count"] = result.get("updated_count", 0)
            summary["updated_paths"] = result.get("updated_paths", [])
        elif operation == "CREATE":
            summary["created_path"] = result.get("created_path")
        
        return summary
    
    def get_session_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all sessions with basic info.
        
        Returns:
            List of session info dictionaries
        """
        sessions = []
        
        for session_dir in sorted(self.base_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name.startswith("Session"):
                continue
            
            session_info = {
                "session_id": session_dir.name,
                "path": str(session_dir)
            }
            
            # Try to load session summary
            summary_file = session_dir / "session_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, "r") as f:
                        summary = json.load(f)
                        session_info.update(summary)
                except Exception:
                    pass
            
            # Count query folders
            query_count = len([
                d for d in session_dir.iterdir() 
                if d.is_dir() and d.name.startswith("query")
            ])
            session_info["query_count"] = query_count
            
            sessions.append(session_info)
        
        return sessions
