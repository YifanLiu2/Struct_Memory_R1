"""
Semantic XPath Orchestrator - Orchestrates the full CRUD pipeline with 2-stage query processing.

Pipeline Stages:
1. Version Resolution (LLM Call 1): Determines version selector and CRUD operation
2. XPath Generation (LLM Call 2): Generates tree-traversal semantic XPath query
3. XPath Execution: Runs semantic XPath to retrieve candidate nodes
4. Downstream Task (LLM Call 3): Single-LLM handler for CRUD-specific processing

Coordinates:
- Version resolution from in-tree versioning
- Semantic XPath query execution
- CRUD-specific handlers (Read, Delete, Update, Create)
- In-tree version management

Tree modifications are delegated to the CRUD handlers.
Includes stage-by-stage timing and token usage tracking.
"""

import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_execution.query_generation.version_crud_resolver.version_resolver import VersionResolver
from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import VersionSelector, ResolvedVersion
from pipeline_execution.query_generation.semantic_xpath_query_generator.xpath_query_generator import XPathQueryGenerator
from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import CRUDOperation
from pipeline_execution.query_generation.semantic_xpath_query_generator.semantic_xpath_query_generator_model import ParsedQuery
from pipeline_execution.semantic_xpath_execution.query_display import canonicalize_query
from pipeline_execution.pipeline_orchestrator.orchestrator_models import PipelineTimer
from pipeline_execution.semantic_xpath_execution import DenseXPathExecutor, get_versioning_info
from utils.tree_modification import VersionManager, copy_version_content
from utils.logger.query_orchestrator_logging import PipelineSummaryLogger

from pipeline_execution.crud.read_handler import ReadHandler
from pipeline_execution.crud.delete_handler import DeleteHandler
from pipeline_execution.crud.update_handler import UpdateHandler
from pipeline_execution.crud.create_handler import CreateHandler
from pipeline_execution.crud.base import HandlerResult


logger = logging.getLogger(__name__)


class SemanticXPathOrchestrator:
    """
    Main orchestrator for CRUD operations with 3-stage LLM processing.
    
    Pipeline:
    1. Version Resolution (LLM) - determines version selector and CRUD operation
    2. XPath Generation (LLM) - generates tree-traversal query
    3. XPath Execution - runs semantic XPath (non-LLM)
    4. Downstream Handler (LLM) - single-call CRUD-specific processing
    5. Tree Modification (if applicable)
    6. Version Creation (if applicable)
    
    Trees are modified in-place with versions stored as child nodes.
    """
    
    def __init__(
        self,
        scoring_method: str = None,
        top_k: int = None,
        score_threshold: float = None,
        tree_path: Path = None,
        traces_path: Path = None,
        config: dict = None
    ):
        """
        Initialize the CRUD executor.
        
        Args:
            scoring_method: Scoring method for semantic XPath
            top_k: Number of top results to consider
            score_threshold: Minimum score threshold
            tree_path: Optional path to XML tree (uses config default if not provided)
            traces_path: Optional directory for trace files
            config: Optional config dict. If not provided, loads from config.yaml.
        """
        # Resolve schema_name from config so all components use the same schema.
        # Config must be provided by the caller (e.g. experiment YAML or CLI).
        if config is None:
            raise ValueError(
                "No config dict provided to SemanticXPathOrchestrator. "
                "Pass a config dict with at least 'active_schema' and 'openai' settings."
            )
        resolved_schema_name = config.get("active_schema")
        
        # Initialize query processing components with explicit schema and config
        self.version_resolver = VersionResolver(schema_name=resolved_schema_name, config=config)
        self.query_generator = XPathQueryGenerator(schema_name=resolved_schema_name, config=config)
        
        # Initialize execution components (handles tree path resolution from config)
        self.executor = DenseXPathExecutor(
            scoring_method=scoring_method,
            top_k=top_k,
            score_threshold=score_threshold,
            tree_path=tree_path,
            traces_path=traces_path,
            config=config,
            schema_name=resolved_schema_name
        )
        
        # Get base directory from executor's resolved tree path
        base_dir = self.executor.memory_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downstream handlers with config for client creation
        schema = self.executor._schema
        handler_traces_path = traces_path / "reasoning_traces" if traces_path else None
        self.read_handler = ReadHandler(schema=schema, traces_path=handler_traces_path, config=config)
        self.delete_handler = DeleteHandler(schema=schema, traces_path=handler_traces_path, config=config)
        self.update_handler = UpdateHandler(schema=schema, traces_path=handler_traces_path, config=config)
        self.create_handler = CreateHandler(schema=schema, traces_path=handler_traces_path, config=config)
        
        # Tree modification components — use resolved_schema_name directly
        # instead of executor.schema_name which could be None when only
        # tree_path was provided without an explicit schema_name
        self.version_manager = VersionManager(
            base_directory=base_dir,
            schema_name=resolved_schema_name
        )
        
        # Store reference to tree for modifications
        self._tree = None
        self._tree = None
        self._schema_name = resolved_schema_name
        self.config = config
    
    @property
    def tree(self) -> ET.ElementTree:
        """Get the current tree (loads from executor if needed)."""
        if self._tree is None:
            self._tree = copy.deepcopy(self.executor.tree)
        return self._tree
    
    @property
    def tree_path(self) -> Path:
        """Get the path to the tree file."""
        return self.executor.memory_path
    
    def _sync_executor_tree(self):
        """Sync the DenseXPathExecutor's tree with our modified tree."""
        self.executor._tree = self._tree
        self.executor._root = self._tree.getroot()
    
    def execute(self, user_query: str) -> Dict[str, Any]:
        """
        Execute a CRUD operation based on the user's query.
        
        Pipeline:
        1. Version Resolution (LLM)
        2. XPath Generation (LLM)
        3. XPath Execution (non-LLM)
        4. Downstream Task Handler (LLM)
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            Dict with operation results, traces, timing info, and token usage
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timer = PipelineTimer()
        
        # Stage 1: Version Resolution (LLM Call 1)
        timer.start("version_resolution")
        version_result = self.version_resolver.resolve(user_query)
        timer.stop(token_usage=version_result.token_usage)
        
        logger.info(f"Version resolved: {version_result.get_version_selector_string()}, {version_result.crud_operation.value}")
        print(f"\n[Version] {version_result.get_version_selector_string()}")
        print(f"[Operation] {version_result.crud_operation.value}")
        
        # Version Lookup (non-LLM)
        timer.start("version_lookup")
        target_version, matched_version = self._resolve_version_element(version_result)
        timer.stop()
        
        if target_version is None:
            return {
                "success": False,
                "error": "No version found in tree",
                "timestamp": timestamp,
                "user_query": user_query,
                "version_resolution": version_result.to_dict(),
                "timing": timer.to_dict()
            }
        
        versioning = get_versioning_info(self._schema_name)
        index_attr = versioning.get("version_index_attr") or "number"
        version_number = target_version.get(index_attr, "?")
        logger.info(f"Operating on version {version_number}")
        print(f"[Target version] {version_number}")
        
        # Extract version change context from the MATCHED version (which triggered selection)
        version_change_context = None
        if matched_version is not None:
            patch_info = matched_version.find("patch_info")
            conv_hist = matched_version.find("conversation_history")
            context_parts = []
            if patch_info is not None and patch_info.text:
                context_parts.append(f"Changes in referenced version: {patch_info.text}")
            if conv_hist is not None and conv_hist.text:
                context_parts.append(f"Original request: {conv_hist.text}")
            
            if context_parts:
                version_change_context = " | ".join(context_parts)
                logger.info(f"Version Context: {version_change_context}")

        # Stage 2: XPath Generation (LLM Call 2)
        # Use task_query (version-stripped query) for xpath generation
        # This prevents the xpath generator from trying to handle version selection
        task_query = version_result.task_query or user_query
        if task_query != user_query:
            print(f"[Task query] {task_query}")
        
        timer.start("query_generation")
        parsed_query = self.query_generator.generate_and_parse(
            task_query, 
            version_result.crud_operation,
            version_change_context=version_change_context
        )
        timer.stop(token_usage=parsed_query.token_usage)
        
        canonical_generated_xpath = canonicalize_query(parsed_query.xpath)
        logger.info(f"Generated XPath (canonical): {canonical_generated_xpath}")
        print(f"[XPath] {canonical_generated_xpath}")
        
        # Stage 3: XPath Execution (non-LLM)
        timer.start("xpath_execution")
        xpath_query = self._build_version_xpath(parsed_query.xpath, target_version)
        execution_result = self.executor.execute(xpath_query)
        timer.stop(token_usage=execution_result.token_usage)
        
        # Get retrieved nodes as dicts
        retrieved_nodes = [m.to_dict() for m in execution_result.matched_nodes]
        print(f"[Retrieved] {len(retrieved_nodes)} candidate nodes")
        
        # Stage 4: Downstream Task Handler (LLM Call 3)
        timer.start("downstream_task")
        
        if version_result.crud_operation == CRUDOperation.READ:
            handler_result = self._execute_read(user_query, retrieved_nodes, parsed_query, version_change_context)
        elif version_result.crud_operation == CRUDOperation.DELETE:
            handler_result = self._execute_delete(user_query, retrieved_nodes, parsed_query, target_version, version_change_context)
        elif version_result.crud_operation == CRUDOperation.UPDATE:
            handler_result = self._execute_update(user_query, retrieved_nodes, parsed_query, target_version, version_change_context)
        elif version_result.crud_operation == CRUDOperation.CREATE:
            handler_result = self._execute_create(user_query, retrieved_nodes, parsed_query, target_version, version_change_context)
        else:
            handler_result = HandlerResult(
                success=False,
                operation="UNKNOWN",
                error=f"Unknown operation: {version_result.crud_operation}"
            )
        
        # Record token usage from handler
        token_usage = handler_result.token_usage.to_dict() if handler_result.token_usage else None
        timer.stop(token_usage=token_usage)
        
        # Build result
        canonical_xpath_query = canonicalize_query(execution_result.query)
        canonical_full_query = f"{handler_result.operation}({canonical_xpath_query})"

        result = {
            "timestamp": timestamp,
            "user_query": user_query,
            "task_query": task_query,  # Version-stripped query used for xpath generation
            "operation": handler_result.operation,
            "success": handler_result.success,
            "version_resolution": version_result.to_dict(),
            "parsed_query": parsed_query.to_dict(),
            "version_used": version_number,
            "xpath_execution": {
                "query": execution_result.query,
                "canonical_query": canonical_xpath_query,
                "matched_count": len(execution_result.matched_nodes),
                "execution_time_ms": execution_result.execution_time_ms,
                "token_usage": execution_result.token_usage
            },
            "handler_result": handler_result.to_dict(),
            "timing": timer.to_dict(),
            # Include visualization data from execution result
            "xpath_query": execution_result.query,
            "full_query": f"{handler_result.operation}({execution_result.query})",
            "canonical_xpath_query": canonical_xpath_query,
            "canonical_full_query": canonical_full_query,
            "traversal_steps": execution_result.traversal_steps,
            "score_fusion_trace": execution_result.score_fusion_trace,
            "final_filtering_trace": execution_result.final_filtering_trace,
            "demo_logger_trace": execution_result.demo_logger_trace,
        }
        
        # Flatten operation-specific fields to top level for backward compatibility
        self._flatten_handler_result(result, handler_result, retrieved_nodes)
        
        if handler_result.error:
            result["error"] = handler_result.error
        
        # Print timing summary
        PipelineSummaryLogger.print_summary(timer)
        
        return result
    
    def _flatten_handler_result(
        self,
        result: Dict[str, Any],
        handler_result: HandlerResult,
        retrieved_nodes: List[Dict[str, Any]]
    ):
        """
        Flatten handler result fields to top level for backward compatibility.
        
        This ensures the result dict has the same structure as the old system,
        so TraceWriter and other consumers can find the expected fields.
        """
        if not handler_result.output:
            return
        
        operation = handler_result.operation
        output = handler_result.output
        
        if operation == "READ":
            # Flatten ReadResult fields
            selected_nodes = []
            if hasattr(output, 'selected_nodes'):
                for node in output.selected_nodes:
                    selected_nodes.append({
                        **node.node_data,
                        "tree_path": node.tree_path,
                        "reasoning": node.reasoning
                    })
            
            result["candidates_count"] = len(retrieved_nodes)
            result["selected_count"] = len(selected_nodes)
            result["selected_nodes"] = selected_nodes
            
        elif operation == "DELETE":
            # Flatten DeleteResult fields
            result["deleted_count"] = len(output.nodes_to_delete) if hasattr(output, 'nodes_to_delete') else 0
            result["deleted_paths"] = output.nodes_to_delete if hasattr(output, 'nodes_to_delete') else []
            
        elif operation == "UPDATE":
            # Flatten UpdateResult fields
            updated_paths = []
            update_results = []
            if hasattr(output, 'updates'):
                for update in output.updates:
                    updated_paths.append(update.tree_path)
                    update_results.append({
                        "path": update.tree_path,
                        "changes": update.changes,
                        "success": True
                    })
            
            result["updated_count"] = len(updated_paths)
            result["updated_paths"] = updated_paths
            result["update_results"] = update_results
            
        elif operation == "CREATE":
            # Flatten CreateResult fields
            result["created_path"] = f"{output.parent_path}/{output.node_type}" if output.parent_path else None
            result["insertion_point"] = {
                "parent_path": output.parent_path,
                "position": output.position,
                "reasoning": output.reasoning
            }
            result["content_result"] = {
                "success": output.created_content is not None,
                "node_type": output.node_type,
                "fields": output.fields
            }

    def _resolve_version_element(self, version_result: ResolvedVersion) -> Tuple[Optional[ET.Element], Optional[ET.Element]]:
        """
        Resolve the actual version element from the version resolution result.
        
        Returns:
            Tuple of (target_version, matched_version)
            - target_version: The version to operate on (after applying 'before' selector)
            - matched_version: The version that matched the selector (contains the context)
        """
        if version_result.semantic_query:
            matched_version = self.version_manager.get_version_by_semantic(
                self.tree,
                version_result.semantic_query,
                scorer=self.executor.scorer
            )
        else:
            matched_version = self.version_manager.get_version_by_number(
                self.tree,
                version_result.index
            )
        
        if version_result.selector_type == VersionSelector.BEFORE and matched_version is not None:
            return self.version_manager.get_previous_version(self.tree, matched_version), matched_version
        
        return matched_version, matched_version
    
    def _execute_read(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        version_change_context: Optional[str] = None
    ) -> HandlerResult:
        """Execute a READ operation using the ReadHandler."""
        return self.read_handler.process(
            user_query,
            retrieved_nodes,
            operation_context={
                "parsed_query": parsed_query.to_dict(),
                "version_change_context": version_change_context
            }
        )
    
    def _execute_delete(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        target_version: ET.Element,
        version_change_context: Optional[str] = None
    ) -> HandlerResult:
        """Execute a DELETE operation using the DeleteHandler."""
        # Get handler decision
        handler_result = self.delete_handler.process(
            user_query,
            retrieved_nodes,
            operation_context={
                "parsed_query": parsed_query.to_dict(),
                "version_change_context": version_change_context
            }
        )
        
        if not handler_result.success or not handler_result.output:
            return handler_result
        
        if not handler_result.output.nodes_to_delete:
            handler_result.success = False
            handler_result.error = "No nodes selected for deletion"
            return handler_result
        
        # Apply tree modifications via handler
        version_content = copy_version_content(target_version, schema_name=self._schema_name, config=self.config)
        mod_result = self.delete_handler.apply_to_content(
            handler_result, version_content, target_version
        )
        
        if not mod_result.success:
            handler_result.success = False
            handler_result.error = mod_result.error
            return handler_result
        
        # Create new version
        self._create_new_version(
            target_version,
            mod_result.modified_content,
            mod_result.patch_info,
            user_query
        )
        
        # Update result with actual deletions
        handler_result.output.nodes_to_delete = mod_result.affected_paths
        return handler_result
    
    def _execute_update(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        target_version: ET.Element,
        version_change_context: Optional[str] = None
    ) -> HandlerResult:
        """Execute an UPDATE operation using the UpdateHandler."""
        # Get handler decision with updates
        handler_result = self.update_handler.process(
            user_query,
            retrieved_nodes,
            operation_context={
                "parsed_query": parsed_query.to_dict(),
                "update_info": parsed_query.update_info,
                "version_change_context": version_change_context
            }
        )
        
        if not handler_result.success or not handler_result.output:
            return handler_result
        
        if not handler_result.output.updates:
            handler_result.success = False
            handler_result.error = "No nodes selected for update"
            return handler_result
        
        # Apply tree modifications via handler
        version_content = copy_version_content(target_version, schema_name=self._schema_name, config=self.config)
        mod_result = self.update_handler.apply_to_content(
            handler_result, version_content, target_version
        )
        
        if not mod_result.success:
            handler_result.success = False
            handler_result.error = mod_result.error
            return handler_result
        
        # Create new version
        self._create_new_version(
            target_version,
            mod_result.modified_content,
            mod_result.patch_info,
            user_query
        )
        
        return handler_result
    
    def _execute_create(
        self,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        parsed_query: ParsedQuery,
        target_version: ET.Element,
        version_change_context: Optional[str] = None
    ) -> HandlerResult:
        """Execute a CREATE operation using the CreateHandler."""
        # Extract create info from parsed query
        create_info = {}
        if parsed_query.create_info:
            parent_path, node_type, description = parsed_query.create_info
            create_info = {
                "parent_path": parent_path,
                "node_type": node_type,
                "description": description
            }
        
        # Get handler decision with created content
        handler_result = self.create_handler.process(
            user_query,
            retrieved_nodes,
            operation_context={
                "parsed_query": parsed_query.to_dict(),
                "create_info": create_info,
                "version_change_context": version_change_context
            }
        )
        
        if not handler_result.success or not handler_result.output:
            return handler_result
        
        if not handler_result.output.created_content:
            handler_result.success = False
            handler_result.error = "Content generation failed"
            return handler_result
        
        # Apply tree modification via handler
        version_content = copy_version_content(target_version, schema_name=self._schema_name, config=self.config)
        mod_result = self.create_handler.apply_to_content(
            handler_result, version_content, target_version
        )
        
        if not mod_result.success:
            handler_result.success = False
            handler_result.error = mod_result.error
            return handler_result
        
        # Create new version
        self._create_new_version(
            target_version,
            mod_result.modified_content,
            mod_result.patch_info,
            user_query
        )
        
        return handler_result
    
    def _create_new_version(
        self,
        source_version: ET.Element,
        modified_content: List[ET.Element],
        patch_info: str,
        user_query: str
    ):
        """
        Create a new version and save the tree.
        
        Args:
            source_version: The source version element
            modified_content: The modified content elements
            patch_info: Description of changes
            user_query: The user's original request
        """
        self.version_manager.create_new_version(
            self.tree,
            source_version,
            patch_info=patch_info,
            conversation_history=user_query,
            modified_content=modified_content
        )
        
        self.version_manager.save_tree(self.tree, self.executor.memory_path)
        self._sync_executor_tree()


    def _build_version_xpath(self, xpath: str, version: ET.Element) -> str:
        """Build an xpath query that targets the specific version.
        
        Handles both regular paths and global index queries:
        - Regular: /Day/POI -> /Root/Itinerary_Version[1]/Day/POI
        - Global: (/Day/POI)[1] -> (/Root/Itinerary_Version[1]/Day/POI)[1]
        
        Note: Uses positional index [N] instead of attribute predicate [@number='N']
        because the semantic XPath parser doesn't support attribute predicates.
        """
        import re
        
        versioning = get_versioning_info(self._schema_name)
        version_tag = versioning.get("version_tag") or version.tag
        index_attr = versioning.get("version_index_attr") or "number"
        version_number = version.get(index_attr) or version.get("number") or "1"
        
        # Build dynamic prefix based on version schema
        path_parts = versioning.get("version_path_parts") or [versioning.get("root_tag"), version_tag]
        path_parts = [p for p in path_parts if p]
        if path_parts:
            prefix_parts = path_parts[:-1] + [f"{version_tag}[{version_number}]"]
            version_prefix = "/" + "/".join(prefix_parts)
            version_path_base = "/" + "/".join(path_parts)
        else:
            version_prefix = f"/{version_tag}[{version_number}]"
            version_path_base = f"/{version_tag}"
        
        def apply_prefix(inner_path: str) -> str:
            if inner_path.startswith(version_prefix):
                return inner_path
            if inner_path.startswith(version_path_base):
                return f"{version_prefix}{inner_path[len(version_path_base):]}"
            if not inner_path.startswith("/"):
                inner_path = "/" + inner_path
            return f"{version_prefix}{inner_path}"
        
        # Handle global index queries: (/path)[index] -> (/version_prefix/path)[index]
        # Pattern matches (/...)[N] or (/...)[N:M] or (/...)[-N:] etc.
        global_index_match = re.match(r'^\((.+)\)(\[.+\])$', xpath)
        if global_index_match:
            inner_path = global_index_match.group(1)
            global_index = global_index_match.group(2)
            return f"({apply_prefix(inner_path)}){global_index}"
        
        # Regular queries
        return apply_prefix(xpath)
    
    def reload_tree(self):
        """Reload the tree from the original file."""
        self._tree = None
        self.executor._tree = None
        self.executor._root = None
