"""
Version Resolver - First stage LLM call for 2-stage semantic XPath processing.

Responsibilities:
1. Determine version selector type (at/before)
2. Extract semantic query for version matching
3. Classify CRUD operation

Query Syntax Output:
- at([-1]) - Latest version (default)
- at([N]) - Specific version number
- at(sem(content ~= "description")) - Semantic match for specific version
- before(sem(content ~= "description")) - Version before the matched version (for rollback)
"""

import re
import logging
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_default_client
from pipeline_execution.semantic_xpath_execution import get_schema_info
from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import CRUDOperation
from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import VersionSelector
from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import ResolvedVersion


logger = logging.getLogger(__name__)


class VersionResolver:
    """
    Resolves version selectors and CRUD operations from natural language queries.
    
    First stage of 2-stage semantic XPath processing:
    1. Version Resolution (this class) - determines which version to operate on
    2. XPath Generation - generates the tree traversal query
    """
    
    # Pattern to parse LLM response (first line: version selector and operation)
    RESPONSE_PATTERN = re.compile(
        r'(at|before)\s*\(\s*'
        r'(?:'
        r'\[\s*(-?\d+)\s*\]'  # Index: [-1] or [2]
        r'|'
        r'sem\s*\(\s*content\s*~=\s*["\']([^"\']+)["\']\s*\)'  # Semantic: sem(content ~= "...")
        r')\s*\)'
        r'\s*,\s*'
        r'(READ|CREATE|UPDATE|DELETE)',
        re.IGNORECASE
    )
    
    # Pattern to parse task query (second line)
    TASK_PATTERN = re.compile(r'task:\s*(.+)', re.IGNORECASE)
    
    def __init__(self, client=None, schema_name: Optional[str] = None, config: dict = None):
        """
        Initialize the version resolver.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created lazily.
            schema_name: Optional schema name. If None, uses active_schema from config.
            config: Optional config dict. Used for schema info and client creation.
        """
        self._client = client
        self._system_prompt = None
        self._config = config
        
        # Get schema info to determine prompt path
        schema_info = get_schema_info(schema_name, config=config)
        prompt_file = schema_info.get(
            "version_resolver_prompt",
            "prompts/query_generator/version_resolver.txt",
        )
        self._prompt_path = Path(__file__).parent.parent.parent.parent / "storage" / prompt_file
        self._schema_name = schema_info["schema_name"]
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from client import OpenAIClient
            self._client = OpenAIClient(config=self._config)
        return self._client
    
    @property
    def system_prompt(self) -> str:
        """Lazy load the system prompt from file."""
        if self._system_prompt is None:
            with open(self._prompt_path, "r", encoding="utf-8") as f:
                self._system_prompt = f.read()
        return self._system_prompt
    
    def resolve(self, user_query: str) -> ResolvedVersion:
        """
        Resolve version selector and CRUD operation from user query.
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            ResolvedVersion with selector type, semantic query or index, CRUD operation,
            and task_query (the portion of the query relevant to xpath generation)
        """
        prompt = f"User: {user_query}"
        
        result = self.client.complete_with_usage(
            prompt,
            system_prompt=self.system_prompt.format(schema_name=self._schema_name),
            temperature=0.1,
            max_tokens=1024
        )
        
        raw_response = result.content.strip()
        
        # Clean up response
        if raw_response.lower().startswith("output:"):
            raw_response = raw_response[7:].strip()
        
        resolved_version = self._parse_response(raw_response, user_query)
        resolved_version.token_usage = result.usage.to_dict()
        return resolved_version
    
    def _parse_response(self, response: str, original_query: str) -> ResolvedVersion:
        """
        Parse the LLM response into a ResolvedVersion.
        
        Expected format: 
            at([-1]), READ
            task: find museums
            
            OR
            
            before(sem(content ~= "delete museum")), UPDATE
            task: update the museums to chinese
        
        Args:
            response: Raw LLM response
            original_query: Original user query (used as fallback for task_query)
            
        Returns:
            ResolvedVersion object
        """
        match = self.RESPONSE_PATTERN.search(response)
        
        # Extract task query
        task_match = self.TASK_PATTERN.search(response)
        task_query = task_match.group(1).strip() if task_match else original_query
        
        if match:
            selector_str = match.group(1).lower()
            index_str = match.group(2)
            semantic_str = match.group(3)
            crud_str = match.group(4).upper()
            
            selector_type = VersionSelector.AT if selector_str == "at" else VersionSelector.BEFORE
            
            if index_str:
                index = int(index_str)
                semantic_query = None
            else:
                index = None
                semantic_query = semantic_str
            
            try:
                crud_operation = CRUDOperation(crud_str.capitalize())
            except ValueError:
                crud_operation = CRUDOperation.READ
            
            return ResolvedVersion(
                selector_type=selector_type,
                semantic_query=semantic_query,
                index=index,
                crud_operation=crud_operation,
                raw_response=response,
                task_query=task_query
            )
        
        # Fallback: default to latest version and try to infer CRUD from keywords
        crud_operation = self._infer_crud_from_text(response)
        
        return ResolvedVersion(
            selector_type=VersionSelector.AT,
            semantic_query=None,
            index=-1,
            crud_operation=crud_operation,
            raw_response=response,
            task_query=task_query
        )
    
    def _infer_crud_from_text(self, text: str) -> CRUDOperation:
        """
        Infer CRUD operation from text using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            CRUDOperation
        """
        text_lower = text.lower()
        
        # Check for CRUD keywords
        delete_keywords = ["delete", "remove", "drop", "cancel", "eliminate"]
        create_keywords = ["add", "create", "insert", "new", "schedule", "put", "include"]
        update_keywords = ["change", "update", "modify", "edit", "move", "reschedule", "replace"]
        
        for keyword in delete_keywords:
            if keyword in text_lower:
                return CRUDOperation.DELETE
        
        for keyword in create_keywords:
            if keyword in text_lower:
                return CRUDOperation.CREATE
        
        for keyword in update_keywords:
            if keyword in text_lower:
                return CRUDOperation.UPDATE
        
        return CRUDOperation.READ
