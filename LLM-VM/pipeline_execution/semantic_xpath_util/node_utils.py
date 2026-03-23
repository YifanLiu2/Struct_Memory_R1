"""
Node Utilities - Helper functions for working with XML Element nodes.

Fully dynamic implementation that works with any tree structure.
No hardcoded node type names - uses structural analysis instead.

Supports schema-aware field lookup when instantiated with a schema,
or falls back to default field names for backwards compatibility.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_execution.semantic_xpath_execution.execution_models import MatchedNode


class NodeUtils:
    """
    Utility class for XML node operations.
    
    Provides methods for:
    - Getting node descriptions and names
    - Extracting subtree information
    - Converting nodes to dictionaries
    
    All methods are dynamic and work with any tree structure by analyzing
    node structure rather than checking specific node type names.
    
    Can be instantiated with a schema for dynamic field lookup per node type,
    or used with static methods for backwards compatibility.
    """
    
    # Default field names to check (in priority order) - used as fallback
    DEFAULT_NAME_FIELDS = ("name", "title", "label")
    DEFAULT_DESC_FIELDS = ("description", "desc", "summary", "content")
    
    # Class-level aliases for backwards compatibility
    NAME_FIELDS = DEFAULT_NAME_FIELDS
    DESC_FIELDS = DEFAULT_DESC_FIELDS
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize NodeUtils with optional schema for dynamic field lookup.
        
        Args:
            schema: Full schema dict with 'nodes' containing node type definitions.
                    Each node type can have a 'fields' list defining available fields.
        """
        self.schema = schema or {}
        self._node_configs: Dict[str, Dict[str, Any]] = self.schema.get("nodes", {})
    
    def _get_name_field_for_node(self, node_tag: str) -> Optional[str]:
        """
        Get the name field for a node type from schema.
        
        Searches the node's schema-defined fields for one that matches
        the known name field patterns.
        
        Args:
            node_tag: The XML tag name of the node type
            
        Returns:
            The field name to use for name, or None if not found in schema
        """
        node_config = self._node_configs.get(node_tag, {})
        fields = node_config.get("fields", [])
        for f in fields:
            if f in self.DEFAULT_NAME_FIELDS:
                return f
        return None
    
    def _get_desc_field_for_node(self, node_tag: str) -> Optional[str]:
        """
        Get the description field for a node type from schema.
        
        Searches the node's schema-defined fields for one that matches
        the known description field patterns.
        
        Args:
            node_tag: The XML tag name of the node type
            
        Returns:
            The field name to use for description, or None if not found in schema
        """
        node_config = self._node_configs.get(node_tag, {})
        fields = node_config.get("fields", [])
        for f in fields:
            if f in self.DEFAULT_DESC_FIELDS:
                return f
        return None
    
    def get_field_value(self, node: ET.Element, field_type: str) -> str:
        """
        Get field value using schema-defined field or fallback to defaults.
        
        Instance method that uses schema when available.
        
        Args:
            node: XML element to extract field from
            field_type: Either "name" or "desc" to indicate which field type
            
        Returns:
            The field value, or empty string if not found
        """
        if field_type == "name":
            # Try schema-specific field first
            schema_field = self._get_name_field_for_node(node.tag)
            if schema_field:
                elem = node.find(schema_field)
                if elem is not None and elem.text:
                    return elem.text
            # Fallback to default fields
            for field in self.DEFAULT_NAME_FIELDS:
                elem = node.find(field)
                if elem is not None and elem.text:
                    return elem.text
        elif field_type == "desc":
            # Try schema-specific field first
            schema_field = self._get_desc_field_for_node(node.tag)
            if schema_field:
                elem = node.find(schema_field)
                if elem is not None and elem.text:
                    return elem.text
            # Fallback to default fields
            for field in self.DEFAULT_DESC_FIELDS:
                elem = node.find(field)
                if elem is not None and elem.text:
                    return elem.text
        return ""
    
    def get_name(self, node: ET.Element) -> str:
        """
        Get the display name of a node (schema-aware instance method).
        
        For container nodes with index/number attribute, uses "{NodeType} {index}" format.
        For nodes with @name attribute, uses the attribute value.
        For leaf nodes, uses schema-defined name field or falls back to defaults.
        """
        # For container nodes with index or number, use "{Tag} {index}" format
        index = node.get("index") or node.get("number")
        if index is not None:
            return f"{node.tag} {index}"
        
        # Check for @name attribute (like Category name="Work")
        name_attr = node.get("name")
        if name_attr:
            return name_attr
        
        # For leaf nodes, use schema-aware field lookup (child element)
        name = self.get_field_value(node, "name")
        if name:
            return name
        
        # Fallback to tag name
        return node.tag
    
    def get_unique_child_name(self, child: ET.Element, parent: ET.Element) -> str:
        """
        Get a unique display name for a child node within its parent context.
        
        When get_name() returns just the tag name (i.e., no @name, @index, or
        name field), and the parent has multiple children of the same tag type,
        appends a 1-based positional index to disambiguate:
          - "Task 1", "Task 2" for sibling Tasks
          - "POI 1", "POI 2" for sibling POIs
        
        When the name is already unique (e.g., "Weekly Groceries", "Day 1"),
        or there's only one sibling of that tag, returns the name as-is.
        
        Args:
            child: The child XML element
            parent: The parent XML element containing the child
            
        Returns:
            Unique display name for the child within the parent
        """
        name = self.get_name(child)
        
        # If get_name returned something other than just the tag,
        # the name is already distinctive (e.g., "Weekly Groceries", "Day 1")
        if name != child.tag:
            return name
        
        # Name is just the tag (e.g., "Task"). Check if there are multiple
        # same-tag siblings that would be ambiguous.
        same_tag_siblings = [c for c in parent if c.tag == child.tag]
        
        if len(same_tag_siblings) <= 1:
            # Only one child of this type — no ambiguity
            return name
        
        # Multiple same-tag siblings: append 1-based index
        for idx, sibling in enumerate(same_tag_siblings, 1):
            if sibling is child:
                return f"{child.tag} {idx}"
        
        # Shouldn't reach here, but fall back to just the tag
        return name
    
    def get_path(self, node: ET.Element) -> str:
        """
        Get a simple path representation for a node.
        
        Note: This returns just the node name as a path identifier.
        The full tree path is maintained by the executor during traversal.
        This method is used for logging/debugging purposes.
        """
        return self.get_name(node)
    
    def get_description(self, node: ET.Element) -> str:
        """
        Get the description of a node (schema-aware instance method).
        
        For container nodes without explicit description, creates summary from children.
        """
        # First, try to find an explicit description field
        desc = self.get_field_value(node, "desc")
        if desc:
            return desc
        
        # For container nodes, generate summary from children
        if self._is_container_node(node):
            children_names = []
            for child in node:
                if self._is_structured_node(child):
                    child_name = self.get_field_value(child, "name")
                    if child_name:
                        children_names.append(child_name)
            if children_names:
                return f"{node.tag} with: {', '.join(children_names[:3])}"
        
        return ""
    
    @staticmethod
    def _get_field_value(node: ET.Element, field_names: Tuple[str, ...]) -> str:
        """Try multiple field names and return the first found value."""
        for field in field_names:
            elem = node.find(field)
            if elem is not None and elem.text:
                return elem.text
        return ""
    
    @staticmethod
    def _is_simple_list(node: ET.Element) -> bool:
        """
        Check if a node is a simple list (like <highlights>).
        
        A simple list has children that are all text-only AND have the same tag.
        Examples: <highlights><highlight>A</highlight><highlight>B</highlight></highlights>
        
        Structured entities like <Task> have heterogeneous children (description, status, etc.)
        and should NOT be considered simple lists even if all children are text-only.
        """
        if len(node) == 0:
            return False
        # All children must be text-only leaves (no grandchildren)
        if not all(len(child) == 0 for child in node):
            return False
        # Additionally, a simple list has homogeneous children (all same tag)
        # Structured entities have heterogeneous children (different tags)
        child_tags = set(child.tag for child in node)
        return len(child_tags) == 1
    
    @staticmethod
    def _is_structured_node(node: ET.Element) -> bool:
        """
        Check if a node is a structured node (entity like POI, Restaurant, Task).
        
        Structured nodes are containers or leaf entities that have meaningful children.
        Simple text elements and simple lists (like highlights) are NOT structured.
        """
        # Must have children or identifying attribute (index, name)
        if node.get("index") is not None or node.get("name") is not None:
            return True
        
        if len(node) == 0:
            return False
        
        # Exclude simple lists (all children are text-only)
        if NodeUtils._is_simple_list(node):
            return False
        
        return True
    
    @staticmethod
    def _is_container_node(node: ET.Element) -> bool:
        """
        Check if a node is a container (has identifying attribute).
        
        Container nodes group other nodes (like Day, Project, Category, Person).
        They are identified by an index attribute (@index) or a name attribute
        (@name), depending on the schema's index_attr setting.
        """
        return node.get("index") is not None or node.get("name") is not None
    
    @staticmethod
    def get_node_description(node: ET.Element) -> str:
        """
        Get the description of a node.
        
        For container nodes without explicit description, creates summary from children.
        Works with any tree structure by detecting node types dynamically.
        """
        # First, try to find an explicit description field
        desc = NodeUtils._get_field_value(node, NodeUtils.DESC_FIELDS)
        if desc:
            return desc
        
        # For container nodes (nodes with index attr), generate summary from children
        if NodeUtils._is_container_node(node):
            children_names = []
            for child in node:
                if NodeUtils._is_structured_node(child):
                    child_name = NodeUtils._get_field_value(child, NodeUtils.NAME_FIELDS)
                    if child_name:
                        children_names.append(child_name)
            if children_names:
                return f"{node.tag} with: {', '.join(children_names[:3])}"
        
        return ""
    
    @staticmethod
    def get_node_name(node: ET.Element) -> str:
        """
        Get the display name of a node.
        
        For container nodes with index/number attribute, uses "{NodeType} {index}" format.
        For nodes with @name attribute, uses the attribute value.
        For leaf nodes, tries common name fields.
        Works with any tree structure.
        """
        # For container nodes with index or number, use "{Tag} {index}" format
        # Check both 'index' and 'number' attributes (Version uses 'number')
        index = node.get("index") or node.get("number")
        if index is not None:
            return f"{node.tag} {index}"
        
        # Check for @name attribute (like Category name="Work")
        name_attr = node.get("name")
        if name_attr:
            return name_attr
        
        # For leaf nodes, try common name fields (child elements)
        name = NodeUtils._get_field_value(node, NodeUtils.NAME_FIELDS)
        if name:
            return name
        
        # Fallback to tag name
        return node.tag
    
    @staticmethod
    def get_subtree_descriptions(node: ET.Element) -> List[Tuple[str, str, str]]:
        """
        Get descriptions from all structured child nodes.
        
        Returns:
            List of (type, name, description) tuples
        """
        results = []
        
        for child in node:
            # Only include structured nodes (not simple text elements)
            if NodeUtils._is_structured_node(child):
                name = NodeUtils._get_field_value(child, NodeUtils.NAME_FIELDS)
                desc = NodeUtils._get_field_value(child, NodeUtils.DESC_FIELDS)
                results.append((child.tag, name, desc))
        
        return results
    
    @staticmethod
    def node_to_dict(node: ET.Element) -> Dict[str, Any]:
        """
        Convert an XML node to a dictionary (node's own data only).
        
        Includes type, attributes, and leaf child elements as fields.
        Handles nested lists (like highlights) automatically.
        """
        result = {
            "type": node.tag,
            "attributes": dict(node.attrib)
        }
        
        for child in node:
            if len(child) == 0:  # Leaf element (simple text)
                result[child.tag] = child.text
            elif all(len(grandchild) == 0 for grandchild in child):
                # Simple list (all grandchildren are text leaves)
                result[child.tag] = [gc.text for gc in child if gc.text]
        
        return result
    
    @staticmethod
    def get_all_children(node: ET.Element) -> List[Dict[str, Any]]:
        """
        Get all structured children of a node as dictionaries.
        
        Works with any tree structure by detecting structured nodes dynamically.
        """
        children = []
        
        for child in node:
            # Only include structured nodes
            if NodeUtils._is_structured_node(child):
                child_dict = {"type": child.tag}
                
                for elem in child:
                    if len(elem) == 0 and elem.text:
                        # Simple text element
                        child_dict[elem.tag] = elem.text
                    elif len(elem) > 0:
                        # Nested list (like highlights)
                        child_dict[elem.tag] = [gc.text for gc in elem if gc.text]
                
                children.append(child_dict)
        
        return children
    
    def get_fields_for_node(self, node_tag: str) -> List[str]:
        """
        Get the list of fields defined for a node type in the schema.
        
        Args:
            node_tag: The XML tag name of the node type
            
        Returns:
            List of field names defined in schema, or empty list if not found
        """
        node_config = self._node_configs.get(node_tag, {})
        return node_config.get("fields", [])
    
    def node_to_dict_schema_aware(self, node: ET.Element) -> Dict[str, Any]:
        """
        Convert an XML node to a dictionary using schema-defined fields.
        
        Unlike the static node_to_dict, this uses the schema to determine
        which fields to include for each node type.
        
        Args:
            node: XML element to convert
            
        Returns:
            Dictionary with type, attributes, and schema-defined fields
        """
        result = {
            "type": node.tag,
            "attributes": dict(node.attrib)
        }
        
        # Get schema-defined fields for this node type
        schema_fields = self.get_fields_for_node(node.tag)
        
        if schema_fields:
            # Use schema-defined fields
            for field_name in schema_fields:
                elem = node.find(field_name)
                if elem is not None:
                    if len(elem) == 0:
                        # Simple text element
                        result[field_name] = elem.text
                    elif all(len(gc) == 0 for gc in elem):
                        # Simple list (like highlights)
                        result[field_name] = [gc.text for gc in elem if gc.text]
        else:
            # Fallback to dynamic extraction (like static node_to_dict)
            for child in node:
                if len(child) == 0:
                    result[child.tag] = child.text
                elif all(len(grandchild) == 0 for grandchild in child):
                    result[child.tag] = [gc.text for gc in child if gc.text]
        
        return result
    
    def get_full_subtree(self, node: ET.Element) -> List[Dict[str, Any]]:
        """
        Recursively get the full subtree with schema-aware fields.
        
        Unlike get_all_children which only gets immediate children,
        this method recursively includes all descendants with their
        schema-defined fields and nested children.
        
        Args:
            node: XML element to get subtree from
            
        Returns:
            List of child dictionaries, each with 'type', schema fields,
            and 'children' containing their own subtrees recursively
        """
        children = []
        
        for child in node:
            # Only include structured nodes (not simple text or list elements)
            if self._is_structured_node(child):
                # Get schema-aware node data
                child_dict = self.node_to_dict_schema_aware(child)
                
                # Recursively get children's subtrees
                grandchildren = self.get_full_subtree(child)
                if grandchildren:
                    child_dict["children"] = grandchildren
                
                children.append(child_dict)
        
        return children
    
    def node_to_matched(
        self, 
        node: ET.Element, 
        tree_path: str,
        score: float = 1.0,
        context_trace: List[Dict[str, Any]] = None
    ) -> "MatchedNode":
        """
        Convert an XML node to a MatchedNode with tree context, full subtree, and score.
        
        Uses schema-aware field extraction and recursive subtree collection
        to provide complete node information to downstream handlers.
        
        Args:
            node: XML element to convert
            tree_path: Full path in tree (e.g., "Root > Day 1 > POI")
            score: Semantic matching score
            context_trace: List of high-scoring predicates from parent nodes
            
        Returns:
            MatchedNode with node data and full recursive subtree
        """
        from pipeline_execution.semantic_xpath_execution.execution_models import MatchedNode
        
        node_data = self.node_to_dict_schema_aware(node)
        children = self.get_full_subtree(node)
        
        return MatchedNode(
            node_data=node_data,
            tree_path=tree_path,
            children=children,
            score=score,
            context_trace=context_trace or []
        )
    
    @classmethod
    def node_to_matched_basic(
        cls, 
        node: ET.Element, 
        tree_path: str, 
        score: float = 1.0
    ) -> "MatchedNode":
        """
        Convert an XML node to a MatchedNode with direct children only (legacy).
        
        Use node_to_matched() instance method for full subtree support.
        """
        from pipeline_execution.semantic_xpath_execution.execution_models import MatchedNode
        
        node_data = cls.node_to_dict(node)
        children = cls.get_all_children(node)
        
        return MatchedNode(
            node_data=node_data,
            tree_path=tree_path,
            children=children,
            score=score
        )
    
    @classmethod
    def node_to_info_dict(
        cls, 
        node: ET.Element, 
        path: str, 
        score: float
    ) -> Dict[str, Any]:
        """
        Create an info dictionary for tracing/logging (static version).
        
        Note: For schema-aware name lookup, use the instance method to_info_dict().
        """
        return {
            "path": path,
            "name": cls.get_node_name(node),
            "type": node.tag,
            "score": score
        }
    
    def to_info_dict(
        self, 
        node: ET.Element, 
        path: str, 
        score: float
    ) -> Dict[str, Any]:
        """
        Create an info dictionary for tracing/logging (schema-aware instance method).
        
        Uses schema-defined fields to get node name.
        """
        return {
            "path": path,
            "name": self.get_name(node),
            "type": node.tag,
            "score": score,
            "node_id": id(node)
        }


    def build_parent_map(self, root: ET.Element) -> Dict[ET.Element, ET.Element]:
        """
        Build a mapping from child nodes to their parent nodes.
        
        This is useful for tracing paths back from descendants to ancestors.
        
        Args:
            root: The root element to build the map from
            
        Returns:
            Dictionary mapping each child element to its parent element
        """
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent
        return parent_map
    
    def get_path_from_ancestor_to_descendant(
        self,
        ancestor: ET.Element,
        descendant: ET.Element,
        ancestor_path: str,
        parent_map: Dict[ET.Element, ET.Element]
    ) -> str:
        """
        Compute the full path from an ancestor node to a descendant node.
        
        This traces back from the descendant to the ancestor using the parent map,
        collecting the intermediate node names, then builds the forward path.
        
        Args:
            ancestor: The ancestor node (where the path starts)
            descendant: The descendant node (where the path ends)
            ancestor_path: The current path string of the ancestor
            parent_map: Mapping from child nodes to parent nodes
            
        Returns:
            Full path string from ancestor to descendant (e.g., "Root > Day 1 > POI")
        """
        # Collect nodes from descendant back to ancestor
        path_nodes = []
        current = descendant
        
        while current is not None and current is not ancestor:
            path_nodes.append(current)
            current = parent_map.get(current)
        
        # If we didn't reach the ancestor, descendant is not under ancestor
        if current is not ancestor:
            # Fallback: just append descendant name directly
            parent = parent_map.get(descendant)
            if parent is not None:
                return f"{ancestor_path} > {self.get_unique_child_name(descendant, parent)}"
            return f"{ancestor_path} > {self.get_name(descendant)}"
        
        # Reverse to get path from ancestor to descendant
        path_nodes.reverse()
        
        # Build the path string
        path = ancestor_path
        for node in path_nodes:
            parent = parent_map.get(node)
            if parent is not None:
                path = f"{path} > {self.get_unique_child_name(node, parent)}"
            else:
                path = f"{path} > {self.get_name(node)}"
        
        return path
