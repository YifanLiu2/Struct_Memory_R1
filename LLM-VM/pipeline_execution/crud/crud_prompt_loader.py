"""
Prompt Loader - Dynamic prompt composition with domain-specific knowledge.

Composes prompts from:
- Base templates (schema-agnostic structure)
- Shared snippets (positional selection rules, position rules)
- Domain-specific content (node fields, update rules, examples)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads and composes prompts dynamically based on schema/domain.
    
    Directory structure:
        storage/prompts/crud/
        ├── templates/           # Base templates with placeholders
        │   ├── create_handler.txt
        │   ├── read_handler.txt
        │   ├── update_handler.txt
        │   └── delete_handler.txt
        ├── shared/              # Shared snippets
        │   ├── positional_selection.txt
        │   └── position_rules.txt
        └── domains/             # Domain-specific content
            ├── itinerary/
            │   ├── node_fields.txt
            │   ├── update_rules.txt
            │   └── examples/
            │       ├── create_examples.txt
            │       ├── read_examples.txt
            │       └── ...
            └── todolist/
                └── ...
    """
    
    PROMPTS_ROOT = Path(__file__).parent.parent.parent / "storage" / "prompts" / "crud"
    TEMPLATES_PATH = PROMPTS_ROOT / "templates"
    SHARED_PATH = PROMPTS_ROOT / "shared"
    DOMAINS_PATH = PROMPTS_ROOT / "domains"
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt loader.
        
        Args:
            schema: Schema dictionary containing 'name' field for domain identification
        """
        self.schema = schema or {}
        self._domain = self._detect_domain()
        self._cache: Dict[str, str] = {}
    
    def _detect_domain(self) -> str:
        """Detect domain from schema name or prompt_domain override."""
        schema_name = self.schema.get("name", "").lower()
        prompt_domain = self.schema.get("prompt_domain", "").lower()
        available = set(self.get_available_domains())
        
        # Prefer explicit prompt_domain override
        if prompt_domain:
            return prompt_domain
        
        # Use schema name when matching a domain folder
        if schema_name in available:
            return schema_name
        
        # Allow a few common aliases without forcing a default domain
        alias_mapping = {
            "todo": "todolist",
            "task": "todolist",
        }
        alias = alias_mapping.get(schema_name)
        if alias and alias in available:
            return alias
        
        # Fall back to schema name; missing files will be empty
        return schema_name or (next(iter(available)) if available else "")
    
    @property
    def domain(self) -> str:
        """Get the current domain."""
        return self._domain
    
    @property
    def schema_name(self) -> str:
        """Get human-readable schema name for prompts."""
        return (
            self.schema.get("prompt_name")
            or self.schema.get("name")
            or self._domain
        )
    
    def _load_file(self, path: Path) -> str:
        """Load a file's content, using cache."""
        cache_key = str(path)
        if cache_key not in self._cache:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._cache[cache_key] = f.read().strip()
            else:
                self._cache[cache_key] = ""
                logger.debug(f"File not found: {path}")
        return self._cache[cache_key]
    
    def _load_shared(self, name: str) -> str:
        """Load a shared snippet."""
        return self._load_file(self.SHARED_PATH / f"{name}.txt")
    
    def _load_domain_file(self, name: str) -> str:
        """Load a domain-specific file."""
        return self._load_file(self.DOMAINS_PATH / self._domain / f"{name}.txt")
    
    def _load_domain_examples(self, operation: str) -> str:
        """Load domain-specific examples for an operation."""
        return self._load_file(
            self.DOMAINS_PATH / self._domain / "examples" / f"{operation}_examples.txt"
        )
    
    def _load_template(self, handler_type: str) -> str:
        """Load a base template."""
        return self._load_file(self.TEMPLATES_PATH / f"{handler_type}.txt")
    
    def load_prompt(self, handler_type: str) -> str:
        """
        Load and compose a complete prompt for a handler.
        
        Args:
            handler_type: One of 'create_handler', 'read_handler', 
                         'update_handler', 'delete_handler'
            
        Returns:
            Fully composed prompt with all placeholders filled
        """
        # Extract operation name (e.g., 'create_handler' -> 'create')
        operation = handler_type.replace("_handler", "")
        
        # Load base template
        template = self._load_template(handler_type)
        
        if not template:
            raise FileNotFoundError(
                f"Prompt template not found: {self.TEMPLATES_PATH / f'{handler_type}.txt'}"
            )
        
        # Load shared snippets
        positional_selection = self._load_shared("positional_selection")
        position_rules = self._load_shared("position_rules")
        
        # Load domain-specific content
        node_fields = self._load_domain_file("node_fields")
        update_rules = self._load_domain_file("update_rules")
        examples = self._load_domain_examples(operation)
        
        # Compose the prompt
        prompt = template.format(
            schema_name=self.schema_name,
            positional_selection=positional_selection,
            position_rules=position_rules,
            node_fields=node_fields,
            update_rules=update_rules,
            examples=examples,
        )
        
        return prompt
    
    def clear_cache(self):
        """Clear the file cache."""
        self._cache.clear()
    
    @classmethod
    def get_available_domains(cls) -> list:
        """List all available domains."""
        if cls.DOMAINS_PATH.exists():
            return [d.name for d in cls.DOMAINS_PATH.iterdir() if d.is_dir()]
        return []
    
    @classmethod
    def get_domain_files(cls, domain: str) -> Dict[str, bool]:
        """
        Check which files exist for a domain.
        
        Returns:
            Dict mapping file types to existence status
        """
        domain_path = cls.DOMAINS_PATH / domain
        files = {
            "node_fields": (domain_path / "node_fields.txt").exists(),
            "update_rules": (domain_path / "update_rules.txt").exists(),
            "create_examples": (domain_path / "examples" / "create_examples.txt").exists(),
            "read_examples": (domain_path / "examples" / "read_examples.txt").exists(),
            "update_examples": (domain_path / "examples" / "update_examples.txt").exists(),
            "delete_examples": (domain_path / "examples" / "delete_examples.txt").exists(),
        }
        return files
