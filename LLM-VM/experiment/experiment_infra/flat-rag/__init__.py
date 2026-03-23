"""
Flat-RAG Experiment Infrastructure.

Baseline approach that:
1. Flattens XML tree into per-leaf-node documents with parent path context
2. Uses dense retrieval (TAS-B embeddings) to retrieve top-k relevant nodes
3. Feeds retrieved nodes to downstream CRUD handlers (same as semantic_xpath)

This provides a comparison baseline against the semantic XPath approach.
"""
