# Literature Review: Structured Memory for Life-Long Helping Agents

This folder contains the literature review supporting the NeurIPS paper on **structured memory for life-long helping agents**. It positions the work, catalogs related work, baselines, and datasets.

## Positioning (one paragraph)

Our contribution is a **structured memory framework for life-long helping agents** that (a) uses a **native hierarchy** (User → Topics with per-node preferences, people, and projects), (b) models **agent execution as user-triggered** (Task → Turn → UserUtterance → Steps with summaries and content refs for token-efficient retrieval), and (c) supports **path- and semantic-based retrieval** plus token-efficient "go back" to intermediate results. This differs from flat memory (Memory-R1, Mem0, RAG), read-only structured access (Semantic XPath), and graph/task-DAG approaches (TME, AriGraph, SGMem) by combining a life-long topic layer with an explicit user-triggered task layer and ref-based storage.

## Contents

| Document | Description |
|----------|-------------|
| [positioning.md](positioning.md) | Full positioning: life-long + task-level schema, relation to Semantic XPath and Memory-R1, differentiation from other lines of work. |
| [related_work.md](related_work.md) | Categorized related work (surveys, flat/graph memory, RL for memory, generative agents, preference memory) with short summaries and differences from ours. |
| [baselines.md](baselines.md) | Table of baselines to compare: memory type, read/write, training, suggested datasets. |
| [datasets.md](datasets.md) | Table of evaluation datasets: domain, scale, what they measure, primary/secondary/optional recommendation. |
| [references.bib](references.bib) | BibTeX entries for all cited papers (can be merged into `struct_memory_R1_latex/reference.bib`). |

## Quick reference

- **Primary baselines:** Zero-shot, In-Context Memory, Flat RAG, Mem0, Semantic XPath, Memory-R1 (Flat), Struct Memory-R1, SGMem, TiMem.
- **Primary datasets:** LoCoMo, LongMemEval, Semantic XPath domains (Itinerary, To-Do, Meal Kit).
- **Secondary/optional:** ConvoMem, EvolMem, Conversation Chronicles, tool-use benchmarks (GTA, MCPVerse, M3-Bench).
