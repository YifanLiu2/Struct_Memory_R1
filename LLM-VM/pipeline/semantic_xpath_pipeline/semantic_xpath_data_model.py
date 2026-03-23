"""
Semantic XPath Data Models and Result Formatting.

Contains:
- ResultFormatter: Format CRUD operation results for display
- SessionStatistics: Track session metrics and statistics
"""

from typing import Dict, Any, List


class SessionStatistics:
    """
    Track statistics for a semantic XPath session.
    
    Monitors operation counts, success rates, and version creation.
    """
    
    def __init__(self):
        """Initialize session statistics."""
        self.operations = 0
        self.reads = 0
        self.creates = 0
        self.updates = 0
        self.deletes = 0
        self.successes = 0
        self.failures = 0
        self.versions_created = 0
    
    def update(self, result: Dict[str, Any]):
        """
        Update statistics based on operation result.
        
        Args:
            result: CRUD operation result dictionary
        """
        self.operations += 1
        
        operation = result.get("operation", "").upper()
        if operation == "READ":
            self.reads += 1
        elif operation == "CREATE":
            self.creates += 1
        elif operation == "UPDATE":
            self.updates += 1
        elif operation == "DELETE":
            self.deletes += 1
        
        if result.get("success"):
            self.successes += 1
            # Track version creation for non-READ operations
            if operation in ("CREATE", "UPDATE", "DELETE") and result.get("tree_version"):
                self.versions_created += 1
        else:
            self.failures += 1
    
    def get_success_rate(self) -> float:
        """
        Calculate success rate percentage.
        
        Returns:
            Success rate as percentage (0-100)
        """
        if self.operations == 0:
            return 0.0
        return (self.successes / self.operations) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary.
        
        Returns:
            Dictionary with all statistics
        """
        return {
            "operations": self.operations,
            "reads": self.reads,
            "creates": self.creates,
            "updates": self.updates,
            "deletes": self.deletes,
            "successes": self.successes,
            "failures": self.failures,
            "versions_created": self.versions_created,
            "success_rate": self.get_success_rate()
        }


class ResultFormatter:
    """
    Format CRUD operation results for display.
    
    Provides consistent formatting for:
    - Read results with node details
    - Create results with insertion info
    - Update results with change tracking
    - Delete results with deletion confirmation
    """
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """
        Format the result for display.
        
        Args:
            result: CRUD operation result dictionary
            
        Returns:
            Formatted string for display
        """
        lines = []
        
        operation = result.get("operation", "UNKNOWN")
        success = result.get("success", False)
        status_icon = "✅" if success else "❌"
        
        lines.append(f"\n{status_icon} {operation} Operation {'Succeeded' if success else 'Failed'}")
        lines.append("=" * 60)
        
        # Version info
        version_used = result.get("version_used")
        if version_used:
            lines.append(f"📌 Operating on Version: {version_used}")
        
        # Timing
        if "total_time_ms" in result:
            lines.append(f"⏱️  Time: {result['total_time_ms']:.1f}ms")
        
        # Operation-specific formatting
        if operation == "READ":
            lines.extend(self._format_read_result(result))
        elif operation == "DELETE":
            lines.extend(self._format_delete_result(result))
        elif operation == "UPDATE":
            lines.extend(self._format_update_result(result))
        elif operation == "CREATE":
            lines.extend(self._format_create_result(result))
        
        # Tree version info (for modifications)
        tree_version = result.get("tree_version")
        if tree_version:
            lines.append(f"\n📁 Tree saved: {tree_version.get('path', 'N/A')}")
            lines.append(f"   Total versions: {tree_version.get('version', 'N/A')}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _format_read_result(self, result: Dict[str, Any]) -> List[str]:
        """
        Format READ operation results.
        
        Args:
            result: READ operation result
            
        Returns:
            List of formatted lines
        """
        lines = []
        
        candidates = result.get("candidates_count", 0)
        selected = result.get("selected_count", 0)
        lines.append(f"\n📊 Results: {selected} selected from {candidates} candidates")
        
        selected_nodes = result.get("selected_nodes", [])
        if selected_nodes:
            lines.append("\n📋 Selected Nodes:")
            lines.append("-" * 50)
            
            for i, node in enumerate(selected_nodes, 1):
                node_type = node.get("type", "?")
                tree_path = node.get("tree_path", "")
                
                # For container nodes (Day, Version), use tree_path for display
                if tree_path:
                    display_name = tree_path.split(" > ")[-1] if " > " in tree_path else tree_path
                elif node.get("attributes", {}).get("index"):
                    display_name = f"{node_type} {node['attributes']['index']}"
                else:
                    display_name = node.get("name", "Unknown")
                
                lines.append(f"\n[{i}] {display_name}")
                if tree_path:
                    lines.append(f"    📍 Path: {tree_path}")
                
                if node.get("description"):
                    desc = node["description"]
                    if len(desc) > 100:
                        desc = desc[:100] + "..."
                    lines.append(f"    📝 {desc}")
                
                if node.get("time_block"):
                    lines.append(f"    🕐 {node['time_block']}")
                
                if node.get("expected_cost"):
                    lines.append(f"    💰 {node['expected_cost']}")
                
                if node.get("highlights"):
                    highlights = node["highlights"]
                    if isinstance(highlights, list):
                        lines.append(f"    ✨ {', '.join(highlights)}")
                
                # For container nodes, display children subtree
                children = node.get("children", [])
                if children:
                    lines.append(f"    📦 Children ({len(children)}):")
                    for child in children:
                        child_type = child.get("type", "?")
                        child_name = child.get("name", "Unknown")
                        child_desc = child.get("description", "")
                        lines.append(f"        - {child_type}: {child_name}")
                        if child_desc:
                            short_desc = child_desc[:60] + "..." if len(child_desc) > 60 else child_desc
                            lines.append(f"          {short_desc}")
        else:
            lines.append("\n⚠️  No nodes matched the query")
        
        return lines
    
    def _format_delete_result(self, result: Dict[str, Any]) -> List[str]:
        """
        Format DELETE operation results.
        
        Args:
            result: DELETE operation result
            
        Returns:
            List of formatted lines
        """
        lines = []
        
        deleted_count = result.get("deleted_count", 0)
        deleted_paths = result.get("deleted_paths", [])
        
        lines.append(f"\n🗑️  Deleted: {deleted_count} node(s)")
        
        if deleted_paths:
            lines.append("\nDeleted Paths:")
            for path in deleted_paths:
                lines.append(f"  ❌ {path}")
        
        return lines
    
    def _format_update_result(self, result: Dict[str, Any]) -> List[str]:
        """
        Format UPDATE operation results.
        
        Args:
            result: UPDATE operation result
            
        Returns:
            List of formatted lines
        """
        lines = []
        
        updated_count = result.get("updated_count", 0)
        updated_paths = result.get("updated_paths", [])
        
        lines.append(f"\n✏️  Updated: {updated_count} node(s)")
        
        update_results = result.get("update_results", [])
        for update in update_results:
            path = update.get("path", "Unknown")
            success = update.get("success", False)
            icon = "✅" if success else "❌"
            lines.append(f"\n{icon} {path}")
            
            changes_data = update.get("changes", {})
            changes = changes_data.get("changes", {})
            if changes:
                for field, change in changes.items():
                    old_val = change.get("from", "?")
                    new_val = change.get("to", "?")
                    lines.append(f"    {field}: {old_val} → {new_val}")
        
        return lines
    
    def _format_create_result(self, result: Dict[str, Any]) -> List[str]:
        """
        Format CREATE operation results.
        
        Args:
            result: CREATE operation result
            
        Returns:
            List of formatted lines
        """
        lines = []
        
        created_path = result.get("created_path")
        
        if created_path:
            lines.append(f"\n➕ Created: {created_path}")
            
            # Show insertion point
            insertion = result.get("insertion_point", {})
            if insertion:
                lines.append(f"\n📍 Insertion Point:")
                lines.append(f"    Parent: {insertion.get('parent_path', 'Unknown')}")
                lines.append(f"    Position: {insertion.get('position', -1)}")
            
            # Show generated content summary
            content = result.get("content_result", {})
            if content.get("success"):
                fields = content.get("fields", {})
                lines.append(f"\n📄 Generated Content:")
                for key, value in fields.items():
                    if isinstance(value, list):
                        lines.append(f"    {key}: {', '.join(str(v) for v in value[:3])}...")
                    elif len(str(value)) > 50:
                        lines.append(f"    {key}: {str(value)[:50]}...")
                    else:
                        lines.append(f"    {key}: {value}")
        else:
            lines.append("\n⚠️  Creation failed")
            if result.get("message"):
                lines.append(f"    Reason: {result['message']}")
        
        return lines
