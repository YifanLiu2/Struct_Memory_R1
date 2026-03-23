"""
LLM Classifier - Uses LLM to determine if a node matches a semantic query.
Uses full subtree information for parent node evaluation.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import OpenAIClient, get_client


class LLMClassifier:
    """
    Classifies whether a node matches a semantic description query using LLM.
    Uses full subtree for parent node evaluation.
    """
    
    PROMPT_PATH = Path(__file__).parent.parent.parent / "store" / "prompts" / "llm_classifier.txt"
    
    def __init__(self, client: OpenAIClient = None):
        """
        Initialize the classifier.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created.
        """
        self._client = client
        self._prompt_template = None  # Lazy load
    
    @property
    def client(self) -> OpenAIClient:
        """Lazy load the OpenAI client"""
        if self._client is None:
            self._client = get_client()
        return self._client
    
    @property
    def prompt_template(self) -> str:
        """Lazy load the prompt template from file"""
        if self._prompt_template is None:
            with open(self.PROMPT_PATH, "r") as f:
                self._prompt_template = f.read()
        return self._prompt_template
    
    def _build_prompt(self, node_info: str, query: str, subtree_info: str = "") -> str:
        """
        Build the prompt by replacing placeholders.
        
        Args:
            node_info: String representation of the node information
            query: The semantic query to match against
            subtree_info: String describing all children in subtree
            
        Returns:
            Formatted prompt string
        """
        prompt = self.prompt_template
        prompt = prompt.replace("<NODE_INFO>", node_info)
        prompt = prompt.replace("<QUERY>", query)
        
        if subtree_info:
            evidence_text = f"Subtree (all children):\n{subtree_info}"
        else:
            evidence_text = "Subtree: (none - this is a leaf node)"
        prompt = prompt.replace("<SUBTREE_EVIDENCE>", evidence_text)
        
        return prompt
    
    def classify(
        self, 
        node_info: str, 
        query: str, 
        subtree_info: str = ""
    ) -> bool:
        """
        Determine if a node matches the semantic query.
        
        Args:
            node_info: String representation of the node (should include description)
            query: The semantic query to match against
            subtree_info: Full subtree information (all children)
            
        Returns:
            True if the node matches the query, False otherwise
        """
        prompt = self._build_prompt(node_info, query, subtree_info)
        
        response = self.client.complete(
            prompt,
            temperature=0.0,
            max_tokens=10
        )
        
        result = response.strip().lower() == "true"
        return result
    
    def classify_node(
        self, 
        node: dict, 
        query: str,
        full_subtree: list[dict] = None
    ) -> bool:
        """
        Determine if a node dict matches the semantic query.
        
        Args:
            node: Node dictionary with 'description' and other fields
            query: The semantic query to match against
            full_subtree: List of ALL children nodes (for parent evaluation)
            
        Returns:
            True if the node matches the query, False otherwise
        """
        node_info = self._node_to_string(node)
        subtree_info = ""
        
        if full_subtree:
            subtree_info = self._subtree_to_string(full_subtree)
        
        return self.classify(node_info, query, subtree_info)
    
    def _node_to_string(self, node: dict) -> str:
        """Convert a node dict to a string representation."""
        parts = []
        
        if "type" in node:
            parts.append(f"Type: {node['type']}")
        
        if "name" in node:
            parts.append(f"Name: {node['name']}")
        
        if "id" in node:
            parts.append(f"ID: {node['id']}")
        
        # Handle description - could be at top level or in attrs
        description = node.get("description")
        if description is None and "attrs" in node:
            description = node["attrs"].get("description")
        
        if description:
            parts.append(f"Description: {description}")
        else:
            parts.append("Description: (none)")
        
        return "\n".join(parts)
    
    def _subtree_to_string(self, children: list[dict]) -> str:
        """Convert all children nodes to a subtree string."""
        if not children:
            return ""
        
        parts = []
        for i, node in enumerate(children, 1):
            node_type = node.get("type", "unknown")
            name = node.get("name", "unnamed")
            desc = node.get("description", "")[:150]
            parts.append(f"  [{i}] {node_type}: {name}")
            if desc:
                parts.append(f"      Description: {desc}")
        
        return "\n".join(parts)


# Convenience function
def is_node_related(
    node: dict, 
    query: str, 
    full_subtree: list[dict] = None,
    client: OpenAIClient = None
) -> bool:
    """
    Check if a node matches a semantic query.
    
    Args:
        node: Node dictionary
        query: Semantic query to match
        full_subtree: List of all children nodes
        client: Optional OpenAI client
        
    Returns:
        True if the node matches the query
    """
    classifier = LLMClassifier(client)
    return classifier.classify_node(node, query, full_subtree)
