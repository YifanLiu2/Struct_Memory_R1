"""
XPath Query Generator - Uses LLM to convert user requests into XPath-like queries.

Tree hierarchy:
Itinerary (root)
├── Day
│   ├── POI
│   └── Restaurant
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_client


class XPathQueryGenerator:
    """
    Converts natural language user requests into XPath-like queries using LLM.
    
    Tree hierarchy:
    - Itinerary (root)
      - Day
        - POI
        - Restaurant
    """
    
    PROMPT_PATH = (
        Path(__file__).parent.parent
        / "storage"
        / "prompts"
        / "query_generator"
        / "xpath_query_generator.txt"
    )
    
    def __init__(self, client=None):
        """
        Initialize the query generator.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created lazily.
        """
        self._client = client
        self._system_prompt = None  # Lazy load
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            self._client = get_client()
        return self._client
    
    @property
    def system_prompt(self) -> str:
        """Lazy load the system prompt from file."""
        if self._system_prompt is None:
            with open(self.PROMPT_PATH, "r") as f:
                self._system_prompt = f.read()
        return self._system_prompt
    
    def generate(self, user_request: str) -> str:
        """
        Convert a user request into an XPath-like query.
        
        Args:
            user_request: Natural language request from the user
            
        Returns:
            XPath-like query string
        """
        prompt = f"User: {user_request}"
        
        response = self.client.complete(
            prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
            max_tokens=256
        )
        
        # Clean up the response - extract just the query
        query = response.strip()
        
        # Remove any "Output:" prefix if present
        if query.lower().startswith("output:"):
            query = query[7:].strip()
        
        return query
