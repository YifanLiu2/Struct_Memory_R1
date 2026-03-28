# Baselines to Compare

Table of baselines for the NeurIPS paper. Columns: method, memory representation, read/write, training, and suggested datasets. See [references.bib](references.bib) for citations.

| Baseline | Paper / Source | Memory representation | Read | Write | Training | Suggested datasets |
|----------|----------------|------------------------|------|-------|----------|--------------------|
| **Zero-shot (no memory)** | — | None | — | — | — | All (lower bound) |
| **In-Context Memory** | Standard | Full history in prompt | Full context | N/A | — | LoCoMo, LongMemEval, ConvoMem (upper bound on info; lost-in-middle, token cost) |
| **Flat RAG** | Lewis et al. 2020; standard RAG | Flat bank, embedding top-k | Semantic search | Optional append | — | LoCoMo, LongMemEval, Semantic XPath domains |
| **Mem0** | mem0.ai; LOCOMO reports | Dynamic flat, semantic scoring | Relevance/importance/recency | Update, contradiction resolve | Optional | LoCoMo, LongMemEval (compare token use and structure) |
| **Semantic XPath** | Liu et al. 2026 | Domain tree, XPath | Path + semantic query | Read-only | — | Semantic XPath domains (Itinerary, To-Do, Meal Kit); extend to LoCoMo struct |
| **Memory-R1 (Flat)** | Yan et al. 2025 | Flat list, ADD/UPDATE/DELETE | Retrieval + distillation | ADD, UPDATE, DELETE, NOOP | GRPO (two agents) | LoCoMo (flat), LongMemEval |
| **Struct Memory-R1** | Current implementation | Tree (Dialogue → Session → Topic → MemoryEntry) | Path + retrieval | ADD (with parent_id), UPDATE, DELETE, NOOP | GRPO (two agents) | LoCoMo (structured), Semantic XPath domains; extend to life-long + task layer in paper |
| **SGMem** | arXiv:2509.21212 | Sentence-level graph, turn/round/session | Multi-hop graph traversal | Implicit via graph | — | LoCoMo, LongMemEval (if available) |
| **TiMem** | arXiv:2601.02845 | Temporal-hierarchical tree, consolidation | Hierarchical retrieval | Consolidation into persona | — | LoCoMo, LongMemEval (report same benchmarks if comparable) |
| **Generative Agents (memory stream)** | Park et al. 2023 | Stream + reflection | Recency/relevance retrieval | Append, reflect | — | Conceptual baseline; sandbox / long-horizon if available |

## Summary

- **Non-RL:** Zero-shot, In-Context Memory, Flat RAG, Mem0, Semantic XPath, SGMem, TiMem, Generative Agents memory.
- **RL:** Memory-R1 (Flat), Struct Memory-R1 (ours; extend to life-long + task-level schema).
- **Key comparisons:** (1) Struct vs flat: Memory-R1 vs Struct Memory-R1. (2) Structured read: Semantic XPath vs ours (we add life-long + task + refs). (3) Long-term QA: SGMem, TiMem, Mem0 on LoCoMo / LongMemEval. (4) Token efficiency: In-Context vs Mem0 vs ours (ref-based).
