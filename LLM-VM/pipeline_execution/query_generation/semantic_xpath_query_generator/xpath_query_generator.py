"""
XPath Query Generator - Second stage LLM call for 2-stage semantic XPath processing.

Generates tree-traversal semantic XPath queries from natural language.
Version handling and CRUD classification are done in the first stage (VersionResolver).

This generator focuses purely on:
1. Understanding the tree structure
2. Generating appropriate semantic predicates
3. Reasoning about context vs actual query information
"""

import re
from pathlib import Path
import sys
from typing import Optional, Tuple, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_default_client
from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import CRUDOperation
from pipeline_execution.query_generation.semantic_xpath_query_generator.semantic_xpath_query_generator_model import ParsedQuery
from pipeline_execution.semantic_xpath_execution import get_schema_info, get_schema_summary_for_prompt
from pipeline_execution.semantic_xpath_parsing import QueryParseError, PredicateParseError, NodeTestParseError
from pipeline_execution.semantic_xpath_parsing.predicate_ast.tokenizer import TokenizeError

# Prompts directory
_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "storage"
    / "prompts"
    / "query_generator"
)


class XPathQueryGenerator:
    """
    Converts natural language user requests into semantic XPath queries using LLM.
    
    Second stage of 2-stage semantic XPath processing:
    1. Version Resolution (VersionResolver) - determines which version to operate on
    2. XPath Generation (this class) - generates the tree traversal query
    
    This class focuses on tree structure navigation and semantic predicate generation.
    It expects the CRUD operation to be provided externally (from VersionResolver).
    """
    
    # Pattern for Create: /parent/path, NodeType, description
    CREATE_PATTERN = re.compile(r'^(.+?),\s*(\w+),\s*(.+)$', re.DOTALL)
    
    # Pattern for Update: /node/path, field: value
    UPDATE_PATTERN = re.compile(r'^(.+?),\s*(.+)$', re.DOTALL)
    
    def __init__(self, client=None, schema_name: Optional[str] = None, config: dict = None):
        """
        Initialize the query generator.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created lazily.
            schema_name: Optional schema name (e.g., "itinerary", "todolist").
                        If None, uses active_schema from config.
            config: Optional config dict. Used for schema info and client creation.
        """
        self._client = client
        self._system_prompt = None  # Lazy load
        self._config = config
        
        # Get schema info for dynamic prompt composition
        self._schema_info = get_schema_info(schema_name, config=config)
        self._schema_name = self._schema_info["schema_name"]
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from client import OpenAIClient
            self._client = OpenAIClient(config=self._config)
        return self._client
    
    @property
    def schema_name(self) -> str:
        """Return the active schema name."""
        return self._schema_name
    
    @property
    def system_prompt(self) -> str:
        """Lazy load and compose the system prompt from template and components."""
        if self._system_prompt is None:
            self._system_prompt = self._compose_prompt()
        return self._system_prompt
    
    def _compose_prompt(self) -> str:
        """
        Compose the system prompt dynamically from template and components.
        
        Loads:
        1. Master template (xpath_query_generator.txt)
        2. Grammar file (xpath_grammar.txt)
        3. Scenario-specific examples
        4. Schema structure (generated from schema)
        5. Syntax rules (from schema config)
        
        Returns:
            Fully composed system prompt string
        """
        # Load master template
        template_path = _PROMPTS_DIR / "xpath_query_generator.txt"
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        
        # Load grammar file
        grammar_path = _PROMPTS_DIR / "xpath_grammar.txt"
        with open(grammar_path, "r", encoding="utf-8") as f:
            grammar = f.read()
        
        # Load scenario-specific examples
        examples_path = Path(self._schema_info["examples_file"])
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = f.read()
        
        # Generate schema structure from schema definition
        schema_structure = get_schema_summary_for_prompt(self._schema_name)
        
        # Get schema-specific syntax rules
        syntax_rules = self._schema_info.get("syntax_rules", "")
        
        # Substitute placeholders
        prompt = template.format(
            schema_name=self._schema_name,
            schema_structure=schema_structure,
            grammar=grammar,
            examples=examples,
            syntax_rules=syntax_rules
        )
        
        return prompt
    
    def generate(self, user_request: str, operation: CRUDOperation = None, version_change_context: str = None, previous_error: str = None, previous_xpath: str = None) -> Tuple[str, Optional[Dict[str, int]]]:
        """
        Convert a user request into a semantic XPath query.
        
        Args:
            user_request: Natural language request from the user
            operation: Pre-determined CRUD operation (from VersionResolver)
            version_change_context: Optional context about version changes (e.g. patch info)
            previous_error: Optional error message from a previous failed parse attempt
            previous_xpath: Optional XPath query from a previous failed attempt
            
        Returns:
            Tuple of (XPath query string, token_usage dict)
        """
        # Build prompt with context and operation hint
        prompt_parts = []
        
        if version_change_context:
            prompt_parts.append(f"Version Change Context: {version_change_context}")
            prompt_parts.append("Use this context to disambiguate the query if needed (e.g. to identify which item was deleted/modified).")
            
        if operation:
            prompt_parts.append(f"Operation: {operation.value}")
        
        # Add error feedback if this is a retry
        if previous_error and previous_xpath:
            prompt_parts.append("")
            prompt_parts.append("IMPORTANT: The previous XPath query you generated was invalid and could not be parsed.")
            prompt_parts.append(f"Previous invalid XPath: {previous_xpath}")
            prompt_parts.append(f"Parse error: {previous_error}")
            prompt_parts.append("Please generate a corrected XPath query that follows the syntax rules exactly.")
            prompt_parts.append("")
            
        prompt_parts.append(f"User: {user_request}")
        
        prompt = "\n".join(prompt_parts)
        
        result = self.client.complete_with_usage(
            prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
            max_tokens=2048
        )
        
        response = result.content
        
        # Clean up the response - extract just the query
        query = response.strip()
        
        # Remove any "Output:" prefix if present
        if query.lower().startswith("output:"):
            query = query[7:].strip()
        
        # Remove markdown code block if present
        if query.startswith("```"):
            lines = query.split("\n")
            query = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            query = query.strip()
        
        return query, result.usage.to_dict()
    
    def generate_and_parse(
        self, 
        user_request: str, 
        operation: CRUDOperation = None,
        version_change_context: str = None,
        max_retries: int = 3
    ) -> ParsedQuery:
        """
        Generate and parse a query from user request with self-reflection on parse errors.
        
        Args:
            user_request: Natural language request from the user
            operation: Pre-determined CRUD operation (from VersionResolver).
                      If None, will try to infer from the query output.
            version_change_context: Optional context about version changes.
            max_retries: Maximum number of retry attempts if parsing fails (default: 3)
            
        Returns:
            ParsedQuery with operation type, parsed components, and token usage
        """
        # Use provided operation or default to READ
        final_operation = operation if operation else CRUDOperation.READ
        
        # Track total token usage across retries
        total_token_usage = {}
        last_error = None
        last_xpath = None
        
        for attempt in range(max_retries):
            xpath_query = None
            try:
                # Generate XPath query
                xpath_query, token_usage = self.generate(
                    user_request, 
                    operation, 
                    version_change_context,
                    previous_error=last_error,
                    previous_xpath=last_xpath
                )
                
                # Merge token usage
                for key, value in token_usage.items():
                    if key in total_token_usage:
                        total_token_usage[key] += value
                    else:
                        total_token_usage[key] = value
                
                # Validate by attempting to parse (for READ/DELETE operations)
                if final_operation in (CRUDOperation.READ, CRUDOperation.DELETE):
                    self._validate_xpath_syntax(xpath_query)
                
                # Parse the query structure
                parsed_query = self._parse_xpath(xpath_query, final_operation)
                parsed_query.token_usage = total_token_usage
                return parsed_query
                
            except (QueryParseError, PredicateParseError, NodeTestParseError, TokenizeError) as e:
                last_error = str(e)
                last_xpath = xpath_query if xpath_query is not None else "(generation failed)"
                
                if attempt < max_retries - 1:
                    print(f"[XPath Parse Error (attempt {attempt + 1}/{max_retries})] {last_error}")
                    print(f"[Invalid XPath] {last_xpath}")
                    print("[Retrying with error feedback...]")
                    continue
                else:
                    # Final attempt failed, raise the error
                    print(f"[XPath Parse Error (final attempt)] {last_error}")
                    print(f"[Invalid XPath] {last_xpath}")
                    raise
        
        # Should never reach here, but just in case
        raise QueryParseError("Failed to generate valid XPath after all retry attempts")
    
    def _validate_xpath_syntax(self, xpath: str):
        """
        Validate XPath syntax by attempting to parse it.
        
        Args:
            xpath: XPath query string to validate
            
        Raises:
            QueryParseError: If the XPath query structure cannot be parsed
            PredicateParseError: If a predicate within the XPath cannot be parsed
            NodeTestParseError: If a node test cannot be parsed
        """
        # Lazy import to avoid circular dependencies
        from pipeline_execution.semantic_xpath_parsing import get_parser
        
        parser = get_parser()
        parser.parse(xpath)
    
    def _parse_xpath(self, xpath: str, operation: CRUDOperation) -> ParsedQuery:
        """
        Parse an xpath query based on operation type.
        
        Args:
            xpath: XPath query string
            operation: CRUD operation type
            
        Returns:
            ParsedQuery with parsed components
        """
        xpath = xpath.strip()
        full_query = f"{operation.value}({xpath})"
        
        if operation == CRUDOperation.CREATE:
            return self._parse_create(xpath, full_query)
        elif operation == CRUDOperation.UPDATE:
            return self._parse_update(xpath, full_query)
        else:
            # READ or DELETE - just xpath
            return ParsedQuery(
                operation=operation,
                xpath=xpath,
                full_query=full_query
            )
    
    def _parse_create(self, content: str, full_query: str) -> ParsedQuery:
        """
        Parse CREATE query content.
        
        Format: /parent/path, NodeType, description
        """
        match = self.CREATE_PATTERN.match(content)
        if match:
            parent_path = match.group(1).strip()
            node_type = match.group(2).strip()
            description = match.group(3).strip()
            
            return ParsedQuery(
                operation=CRUDOperation.CREATE,
                xpath=parent_path,
                full_query=full_query,
                create_info=(parent_path, node_type, description)
            )
        
        # Fallback: treat entire content as xpath
        return ParsedQuery(
            operation=CRUDOperation.CREATE,
            xpath=content,
            full_query=full_query
        )
    
    def _parse_update(self, content: str, full_query: str) -> ParsedQuery:
        """
        Parse UPDATE query content.
        
        Format: /node/path, field: value (or field: value, field2: value2)
        """
        match = self.UPDATE_PATTERN.match(content)
        if match:
            node_path = match.group(1).strip()
            changes_str = match.group(2).strip()
            
            # Parse changes (field: value pairs)
            changes = {}
            for part in changes_str.split(","):
                if ":" in part:
                    key, value = part.split(":", 1)
                    changes[key.strip()] = value.strip()
            
            return ParsedQuery(
                operation=CRUDOperation.UPDATE,
                xpath=node_path,
                full_query=full_query,
                update_info=(node_path, changes)
            )
        
        # Fallback: treat entire content as xpath
        return ParsedQuery(
            operation=CRUDOperation.UPDATE,
            xpath=content,
            full_query=full_query
        )
