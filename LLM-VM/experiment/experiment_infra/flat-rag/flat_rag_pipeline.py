"""
Flat-RAG Pipeline - Flatten XML to leaf-node documents, dense retrieval, downstream CRUD.

Methodology:
1. Flatten: Extract all leaf nodes from XML, each paired with its full parent path
   from Root > Version > Container > ... > LeafNode. Each leaf node becomes a
   "document" containing all its field values and parent context.

2. Embed: Use TAS-B (or OpenAI) embeddings to embed all documents.

3. Retrieve: Given a user query, compute cosine similarity and retrieve top-k nodes.

4. Downstream: Feed retrieved nodes to existing CRUD handlers (same handlers
   used by the semantic_xpath pipeline) to process READ/CREATE/UPDATE/DELETE.

This acts as a baseline comparison to semantic XPath, replacing the XPath
generation + execution stages with simple dense retrieval.
"""

import time
import logging
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from client import OpenAIClient, get_default_client, TASBClient, get_tas_b_client
from client.embedding_cache import CachingEmbeddingClient, DEFAULT_CACHE_DIR
from pipeline_execution.semantic_xpath_util import (
    load_schema, get_versioning_info, get_data_path, NodeUtils
)
from pipeline_execution.query_generation.version_crud_resolver.version_resolver import VersionResolver
from pipeline_execution.query_generation.version_crud_resolver.version_selector_model import (
    CRUDOperation, ResolvedVersion
)
from pipeline_execution.crud import (
    ReadHandler, DeleteHandler, UpdateHandler, CreateHandler,
    HandlerResult
)
from utils.tree_modification import VersionManager, copy_version_content


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class FlatDocument:
    """A flattened leaf-node document for dense retrieval."""
    tree_path: str            # Full path: "Root > Itinerary_Version 1 > Day 3 > Art Gallery of Ontario"
    document_text: str        # Combined text for embedding
    node_data: Dict[str, Any] # Node dict for downstream handlers
    children: List[Dict]      # Subtree children (for handler format)
    element: ET.Element       # Reference to XML element (for tree modifications)
    score: float = 0.0        # Retrieval similarity score


@dataclass
class FlatRAGResult:
    """Result from flat-rag pipeline processing."""
    success: bool
    operation: str
    version_used: int
    retrieved_docs: List[Dict[str, Any]]  # Top-k retrieved documents
    handler_result: Optional[Dict[str, Any]]
    token_usage: Dict[str, Any]
    execution_time_ms: float
    retrieval_time_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "version_used": self.version_used,
            "retrieved_docs": self.retrieved_docs,
            "handler_result": self.handler_result,
            "token_usage": self.token_usage,
            "execution_time_ms": self.execution_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "error": self.error
        }


class FlatRAGPipeline:
    """
    Pipeline: Flatten XML → Dense Retrieval → Downstream CRUD.

    Replaces the XPath generation + execution stages of semantic_xpath
    with a simple embed-and-retrieve approach over flattened leaf nodes.
    """

    def __init__(
        self,
        tree_path: Path = None,
        top_k: int = 5,
        config: Dict[str, Any] = None,
        traces_path: Path = None
    ):
        """
        Initialize the flat-rag pipeline.

        Args:
            tree_path: Path to the XML tree file
            top_k: Number of top results to retrieve
            config: Experiment config dict (must include active_schema, openai, etc.)
            traces_path: Optional directory for trace files
        """
        if config is None:
            raise ValueError("Config dict is required for FlatRAGPipeline.")

        self._config = config
        self._tree_path = tree_path
        self._top_k = top_k
        self._traces_path = traces_path

        # Load schema and versioning info
        self._schema = load_schema(config=config)
        self._versioning = get_versioning_info(config=config)
        self._schema_name = config.get("active_schema")

        # Node utils for schema-aware operations
        self._node_utils = NodeUtils(schema=self._schema)

        # Identify leaf and container types from schema
        self._node_configs = self._schema.get("nodes", {})
        self._leaf_types = {
            name for name, cfg in self._node_configs.items()
            if cfg.get("type") == "leaf"
        }
        self._container_types = {
            name for name, cfg in self._node_configs.items()
            if cfg.get("type") in ("container", "root")
        }
        
        # Debug logging
        logger.info(f"Schema: {self._schema_name}")
        logger.info(f"Leaf types from schema: {self._leaf_types}")
        logger.info(f"Container types from schema: {self._container_types}")
        logger.info(f"Node configs: {list(self._node_configs.keys())}")

        # Version resolver (reuse existing LLM-based version + CRUD classification)
        self.version_resolver = VersionResolver(
            schema_name=self._schema_name, config=config
        )

        # Version manager
        self.version_manager = VersionManager(schema_name=self._schema_name)

        # Embedding client (TAS-B for local dense retrieval)
        self._embedder: Optional[TASBClient] = None

        # CRUD handlers (same as semantic_xpath pipeline)
        handler_traces_path = traces_path / "reasoning_traces" if traces_path else None
        self.read_handler = ReadHandler(
            schema=self._schema, traces_path=handler_traces_path, config=config
        )
        self.delete_handler = DeleteHandler(
            schema=self._schema, traces_path=handler_traces_path, config=config
        )
        self.update_handler = UpdateHandler(
            schema=self._schema, traces_path=handler_traces_path, config=config
        )
        self.create_handler = CreateHandler(
            schema=self._schema, traces_path=handler_traces_path, config=config
        )

        # Cached flattened documents and embeddings
        self._flat_docs: List[FlatDocument] = []
        self._doc_embeddings: Optional[np.ndarray] = None

        # Parse and flatten the tree
        self._tree: Optional[ET.ElementTree] = None
        if tree_path:
            self._load_and_flatten()

    @property
    def embedder(self):
        """Lazy-load embedding client (with disk cache)."""
        if self._embedder is None:
            delegate = get_tas_b_client()
            self._embedder = CachingEmbeddingClient(
                delegate,
                cache_dir=DEFAULT_CACHE_DIR,
                openai_config=None,
            )
        return self._embedder

    # ----------------------------------------------------------------
    # Tree loading and flattening
    # ----------------------------------------------------------------

    def _load_and_flatten(self):
        """Load tree from file, flatten all leaf nodes, and embed them."""
        self._tree = ET.parse(self._tree_path)
        self._flatten_tree()
        self._embed_documents()

    def reload_tree(self):
        """Reload tree from file and re-flatten."""
        self._load_and_flatten()

    def set_traces_path(self, traces_path: Path):
        """Update trace output path for a new query."""
        self._traces_path = traces_path
        handler_traces_path = traces_path / "reasoning_traces" if traces_path else None
        self.read_handler.traces_path = handler_traces_path
        self.delete_handler.traces_path = handler_traces_path
        self.update_handler.traces_path = handler_traces_path
        self.create_handler.traces_path = handler_traces_path
        if handler_traces_path:
            handler_traces_path.mkdir(parents=True, exist_ok=True)

    def _flatten_tree(self):
        """
        Walk the XML tree and extract all leaf nodes into flat documents.

        For each leaf node:
        - Build full tree path: Root > Itinerary_Version 1 > Day 3 > Art Gallery of Ontario
        - Include the content of every ancestor node from root down to this node
        - Combine all field values into a single document text for embedding
        - Store node data dict (same format CRUD handlers expect)
        """
        root = self._tree.getroot()
        self._flat_docs = []

        # Build parent map for path tracing
        parent_map = self._node_utils.build_parent_map(root)

        # Determine version elements
        version_tag = self._versioning.get("version_tag")

        def walk(element: ET.Element, path_parts: List[str], ancestors: List[ET.Element], parent: ET.Element = None):
            """Recursively walk tree, collecting leaf nodes."""
            # Get display name for this element — use unique name when parent
            # is available so that sibling nodes without a name field (e.g. Task)
            # get disambiguated as "Task 1", "Task 2", etc.
            if parent is not None:
                name = self._node_utils.get_unique_child_name(element, parent)
            else:
                name = self._node_utils.get_name(element)
            current_path = " > ".join(path_parts + [name])

            tag = element.tag

            # Check if this is a leaf type
            if tag in self._leaf_types:
                # Build document text including all ancestor content
                doc_text = self._build_document_text(
                    element, path_parts, name, ancestors
                )

                # Build node data dict (format expected by CRUD handlers)
                node_data = self._node_utils.node_to_dict_schema_aware(element)
                children = self._node_utils.get_full_subtree(element)

                self._flat_docs.append(FlatDocument(
                    tree_path=current_path,
                    document_text=doc_text,
                    node_data=node_data,
                    children=children,
                    element=element
                ))
                return

            # Recurse into children
            for child in element:
                # Skip metadata nodes
                if child.tag in ("patch_info", "conversation_history"):
                    continue
                walk(child, path_parts + [name], ancestors + [element], parent=element)

        # Start from root
        walk(root, [], [])
        logger.info(f"Flattened {len(self._flat_docs)} leaf nodes from tree")
        
        # Debug: log if no leaf nodes found
        if len(self._flat_docs) == 0:
            logger.warning(f"No leaf nodes found! Leaf types: {self._leaf_types}")
            # Try to find what tags actually exist in the tree
            all_tags = set()
            def collect_tags(elem):
                all_tags.add(elem.tag)
                for child in elem:
                    collect_tags(child)
            collect_tags(root)
            logger.warning(f"Tags found in XML tree: {sorted(all_tags)}")

    def _build_document_text(
        self,
        element: ET.Element,
        parent_path_parts: List[str],
        node_name: str,
        ancestors: List[ET.Element] = None
    ) -> str:
        """
        Build a text document for a leaf node by combining all ancestor
        content (from root down) and the leaf node's own field values.

        Each ancestor's attributes and direct text fields are included so
        that the flattened document carries full context.

        Example output:
            "Path: Root > Itinerary_Version 1 > Day 3 > Art Gallery of Ontario
             --- Ancestor: Root ---
             --- Ancestor: Itinerary_Version 1 (number=1) ---
             patch_info: Initial version
             --- Ancestor: Day 3 (index=3) ---
             theme: Culture & Museums
             --- Current Node ---
             Type: POI
             Name: Art Gallery of Ontario
             description: World-class art museum...
             time_block: 12:00 PM - 2:00 PM
             highlights: Indoor museum, Cultural experience"
        """
        parts = []

        # Full path context
        full_path = " > ".join(parent_path_parts + [node_name])
        parts.append(f"Path: {full_path}")

        # Ancestor content (root → immediate parent)
        if ancestors:
            for ancestor in ancestors:
                anc_name = self._node_utils.get_name(ancestor)
                # Include attributes
                attrs = " ".join(f'{k}={v}' for k, v in ancestor.attrib.items())
                header = f"--- Ancestor: {anc_name}"
                if attrs:
                    header += f" ({attrs})"
                header += " ---"
                parts.append(header)

                # Include direct text fields of ancestor (skip child containers/leaves)
                for child in ancestor:
                    # Skip nodes that are containers or leaves themselves
                    if child.tag in self._leaf_types or child.tag in self._container_types:
                        continue
                    # Skip version metadata
                    if child.tag in ("patch_info", "conversation_history"):
                        continue
                    if len(child) == 0 and child.text:
                        parts.append(f"  {child.tag}: {child.text}")
                    elif len(child) > 0:
                        items = [gc.text for gc in child if gc.text]
                        if items:
                            parts.append(f"  {child.tag}: {', '.join(items)}")

        # Current leaf node
        parts.append("--- Current Node ---")
        parts.append(f"Type: {element.tag}")
        parts.append(f"Name: {node_name}")

        # All field values from the leaf element
        for child in element:
            if len(child) == 0 and child.text:
                parts.append(f"{child.tag}: {child.text}")
            elif len(child) > 0:
                items = [gc.text for gc in child if gc.text]
                if items:
                    parts.append(f"{child.tag}: {', '.join(items)}")

        return "\n".join(parts)

    def _embed_documents(self):
        """Embed all flattened documents using TAS-B with L2 normalization for cosine similarity."""
        if not self._flat_docs:
            self._doc_embeddings = np.array([])
            return

        texts = [doc.document_text for doc in self._flat_docs]
        # Embed with L2 normalization: required for cosine similarity computation
        self._doc_embeddings = self.embedder.get_embeddings(texts, normalize=True)
        logger.info(f"Embedded {len(texts)} documents, shape: {self._doc_embeddings.shape}")
        
        # Verify embeddings are normalized (for debugging)
        if len(self._doc_embeddings) > 0:
            sample_norm = np.linalg.norm(self._doc_embeddings[0])
            if not np.isclose(sample_norm, 1.0, atol=1e-5):
                logger.warning(f"Document embeddings not normalized: sample norm={sample_norm:.6f}")

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = None) -> List[FlatDocument]:
        """
        Retrieve top-k documents by cosine similarity to the query.

        Args:
            query: User query text
            top_k: Number of results (defaults to self._top_k)

        Returns:
            List of FlatDocument sorted by descending score
        """
        if top_k is None:
            top_k = self._top_k

        if self._doc_embeddings is None or len(self._doc_embeddings) == 0:
            return []

        # Embed query with L2 normalization
        query_embedding = self.embedder.get_embedding(query, normalize=True)

        # Compute cosine similarity: dot product of normalized vectors = cosine similarity
        # Since both query_embedding and doc_embeddings are L2-normalized,
        # np.dot gives us cosine similarity directly (range: -1 to 1, typically 0 to 1 for semantic embeddings)
        scores = np.dot(self._doc_embeddings, query_embedding)
        
        # Verify embeddings are normalized (for debugging)
        if len(query_embedding) > 0:
            query_norm = np.linalg.norm(query_embedding)
            if not np.isclose(query_norm, 1.0, atol=1e-5):
                logger.warning(f"Query embedding not normalized: norm={query_norm:.6f}")

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self._flat_docs[idx]
            doc.score = float(scores[idx])
            results.append(doc)

        return results

    # ----------------------------------------------------------------
    # Main processing
    # ----------------------------------------------------------------

    def process_request(self, user_query: str) -> FlatRAGResult:
        """
        Process a user request through the flat-rag pipeline.

        Pipeline stages:
        1. Version Resolution + CRUD Classification (LLM call)
        2. Dense Retrieval (local embedding similarity)
        3. Downstream CRUD Handler (LLM call)

        Args:
            user_query: Natural language query

        Returns:
            FlatRAGResult with operation details and handler results
        """
        start_time = time.perf_counter()
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_stage": {}
        }

        try:
            # Stage 1: Version Resolution + CRUD Classification
            version_result = self.version_resolver.resolve(user_query)
            task_query = version_result.task_query or user_query
            crud_op = version_result.crud_operation

            if version_result.token_usage:
                token_usage["by_stage"]["version_resolution"] = version_result.token_usage
                token_usage["prompt_tokens"] += version_result.token_usage.get("prompt_tokens", 0)
                token_usage["completion_tokens"] += version_result.token_usage.get("completion_tokens", 0)
                token_usage["total_tokens"] += version_result.token_usage.get("total_tokens", 0)

            # Resolve version number
            version_number = self._resolve_version_number(version_result)

            # Stage 2: Dense Retrieval
            retrieval_start = time.perf_counter()
            retrieved_docs = self.retrieve(task_query)
            retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

            # Convert to handler format (same as semantic_xpath's retrieved_nodes)
            retrieved_nodes = self._docs_to_handler_format(retrieved_docs)

            print(f"[Flat-RAG] Retrieved {len(retrieved_nodes)} nodes (retrieval: {retrieval_time_ms:.0f}ms)")

            # Stage 3: Downstream CRUD Handler
            handler_result = self._execute_downstream(
                crud_op, user_query, retrieved_nodes, version_result
            )

            # Accumulate handler token usage
            if handler_result.token_usage:
                handler_tokens = handler_result.token_usage.to_dict()
                handler_op = handler_result.operation.lower() if handler_result.operation else "handler"
                token_usage["by_stage"][f"{handler_op}_handler"] = handler_tokens
                token_usage["prompt_tokens"] += handler_tokens.get("prompt_tokens", 0)
                token_usage["completion_tokens"] += handler_tokens.get("completion_tokens", 0)
                token_usage["total_tokens"] += handler_tokens.get("total_tokens", 0)

            total_time_ms = (time.perf_counter() - start_time) * 1000

            # Build result
            return FlatRAGResult(
                success=handler_result.success,
                operation=handler_result.operation,
                version_used=version_number,
                retrieved_docs=[
                    {
                        "tree_path": doc.tree_path,
                        "score": doc.score,
                        "node_type": doc.node_data.get("type", ""),
                        "node_name": doc.node_data.get(
                            "name", doc.node_data.get("title", "")
                        ),
                    }
                    for doc in retrieved_docs
                ],
                handler_result=handler_result.to_dict(),
                token_usage=token_usage,
                execution_time_ms=total_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                error=handler_result.error if not handler_result.success else None
            )

        except Exception as e:
            total_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Flat-RAG pipeline error: {e}", exc_info=True)
            return FlatRAGResult(
                success=False,
                operation="UNKNOWN",
                version_used=1,
                retrieved_docs=[],
                handler_result=None,
                token_usage=token_usage,
                execution_time_ms=total_time_ms,
                retrieval_time_ms=0,
                error=str(e)
            )

    # ----------------------------------------------------------------
    # Downstream CRUD dispatch
    # ----------------------------------------------------------------

    def _execute_downstream(
        self,
        crud_op: CRUDOperation,
        user_query: str,
        retrieved_nodes: List[Dict[str, Any]],
        version_result: ResolvedVersion
    ) -> HandlerResult:
        """Dispatch to the appropriate CRUD handler."""
        if crud_op == CRUDOperation.READ or crud_op is None:
            return self.read_handler.process(user_query, retrieved_nodes)

        # For CUD, resolve the target version element for tree modifications
        target_version = self._resolve_version_element(version_result)

        if crud_op == CRUDOperation.DELETE:
            handler_result = self.delete_handler.process(user_query, retrieved_nodes)
            if handler_result.success and handler_result.output and target_version is not None:
                version_content = copy_version_content(
                    target_version, schema_name=self._schema_name, config=self._config
                )
                self.delete_handler.apply_to_content(
                    handler_result, version_content, target_version
                )
            return handler_result

        elif crud_op == CRUDOperation.UPDATE:
            handler_result = self.update_handler.process(user_query, retrieved_nodes)
            if handler_result.success and handler_result.output and target_version is not None:
                version_content = copy_version_content(
                    target_version, schema_name=self._schema_name, config=self._config
                )
                self.update_handler.apply_to_content(
                    handler_result, version_content, target_version
                )
            return handler_result

        elif crud_op == CRUDOperation.CREATE:
            handler_result = self.create_handler.process(
                user_query, retrieved_nodes,
                operation_context={"create_info": {}}
            )
            if handler_result.success and handler_result.output and target_version is not None:
                version_content = copy_version_content(
                    target_version, schema_name=self._schema_name, config=self._config
                )
                self.create_handler.apply_to_content(
                    handler_result, version_content, target_version
                )
            return handler_result

        return HandlerResult(
            success=False, operation="UNKNOWN",
            error=f"Unknown CRUD operation: {crud_op}"
        )

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _docs_to_handler_format(
        self, docs: List[FlatDocument]
    ) -> List[Dict[str, Any]]:
        """
        Convert FlatDocuments to the dict format expected by CRUD handlers.

        This matches the format from DenseXPathExecutor's MatchedNode.to_dict():
            {"node": {...}, "tree_path": "...", "score": 0.95, "children": [...]}
        """
        result = []
        for doc in docs:
            result.append({
                "node": doc.node_data,
                "tree_path": doc.tree_path,
                "score": doc.score,
                "children": doc.children,
            })
        return result

    def _resolve_version_number(self, version_result: ResolvedVersion) -> int:
        """Resolve the version number from version resolution result."""
        if self._tree is None:
            return 1
        if version_result.semantic_query:
            matched = self.version_manager.get_version_by_semantic(
                self._tree, version_result.semantic_query
            )
        else:
            matched = self.version_manager.get_version_by_number(
                self._tree, version_result.index
            )
        if matched is not None:
            idx_attr = self._versioning.get("version_index_attr", "number")
            return int(matched.get(idx_attr, 1))
        return 1

    def _resolve_version_element(
        self, version_result: ResolvedVersion
    ) -> Optional[ET.Element]:
        """Resolve the actual version element from the version resolution result."""
        if self._tree is None:
            return None
        if version_result.semantic_query:
            matched = self.version_manager.get_version_by_semantic(
                self._tree, version_result.semantic_query
            )
        else:
            matched = self.version_manager.get_version_by_number(
                self._tree, version_result.index
            )
        if version_result.selector_type.value == "before" and matched is not None:
            return self.version_manager.get_previous_version(self._tree, matched)
        return matched

    def get_flat_documents(self) -> List[Dict[str, Any]]:
        """
        Return all flattened documents in a JSON-serializable format.

        Each entry contains:
            - tree_path: full path from root
            - document_text: the combined text used for embedding
            - node_type: tag of the element
            - node_name: display name from the element
        """
        docs = []
        for doc in self._flat_docs:
            docs.append({
                "tree_path": doc.tree_path,
                "document_text": doc.document_text,
                "node_type": doc.element.tag if doc.element is not None else "",
                "node_name": doc.node_data.get(
                    "name", doc.node_data.get("title", "")
                ),
            })
        return docs

    def save_tree(self, tree: ET.ElementTree, path: Path):
        """Save modified tree to file."""
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
