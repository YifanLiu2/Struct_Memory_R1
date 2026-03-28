# Related Work

Categorized related work with short summaries and how each differs from our structured memory for life-long helping agents. See [references.bib](references.bib) for full citations.

---

## 1. Long-term and conversational memory for LLM agents (surveys)

**Memory Mechanism of LLM-based Agents** (survey, arXiv:2404.13501; ACM survey 2024)  
Surveys memory in LLM agents: types (episodic, semantic, procedural), operations (storage, retrieval, update), and architectures. **Difference from ours:** Organizing framework only; we contribute a concrete two-layer schema (life-long + task) and user-triggered turn structure with refs.

**Rethinking Memory in LLM based Agents** (arXiv:2505.00675)  
Categorizes memory as parametric vs contextual and six operations (consolidation, updating, indexing, forgetting, retrieval, condensation); discusses long-term, long-context, parametric modification, and multi-source memory. **Difference from ours:** We instantiate a native tree schema and path/semantic retrieval over it, with explicit task-level steps and ref-based token efficiency.

---

## 2. Flat and retrieval-based memory

**RAG** (Lewis et al., NeurIPS 2020)  
Retrieval-augmented generation: retrieve from a corpus and condition generation on retrieved passages. **Difference from ours:** Flat retrieval; we use a structured tree and path/semantic queries over life-long and task layers.

**kNN-LM** (Khandelwal et al., ICLR 2020)  
Nearest-neighbor retrieval over a datastore for language modeling. **Difference from ours:** No explicit schema or hierarchy; we have typed nodes (Topic, Task, Turn, Step) and path-based access.

**Lost in the Middle** (Liu et al., TACL 2024)  
Shows that LMs use information at the start and end of long context better than the middle. **Difference from ours:** Motivates token-efficient retrieval and ref-based loading; we address it via summaries + refs and selective path retrieval instead of dumping full history.

**Mem0** (mem0.ai; LOCOMO benchmark reports)  
Persistent memory layer for LLMs: automatic extraction, dynamic update, contradiction resolution, semantic search with relevance/importance/recency. **Difference from ours:** Flat or semantic-only; we add a native tree (topics, tasks, turns, steps) and path-based retrieval.

**ConvoMem** (arXiv:2511.10523)  
Benchmark (75k+ QA pairs) studying when conversational memory needs RAG; shows simple full-context can reach 70–82% for &lt;150 conversations while some RAG systems lag. **Difference from ours:** We focus on how to structure memory (two-layer + refs) rather than only when to use retrieval.

---

## 3. Structured / graph / hierarchical memory

**Semantic XPath** (Liu et al., 2026)  
XPath-style queries over domain trees (e.g. Itinerary → Day → POI) for conversational AI. **Difference from ours:** Read-only, domain-specific trees; we extend to life-long (topics, preferences, people) and task-level (Turn → UserUtterance → Steps with refs) with writes.

**SGMem** (arXiv:2509.21212)  
Sentence-level graph memory for long-term conversational agents; turn/round/session organization; multi-hop traversal over sentences and generated memory. **Difference from ours:** Sentence-level graph, not a task-level tree; we have explicit Turn/UserUtterance/Step and life-long topic layer with refs.

**TiMem** (arXiv:2601.02845)  
Temporal-hierarchical memory: conversations organized in a temporal tree, raw observations consolidated into abstracted persona representations. **Difference from ours:** We add an explicit task layer (user-triggered turns and steps) and ref-based storage for intermediate results; life-long layer is topic/preference/people, not only temporal consolidation.

**Task Memory Engine (TME)** (arXiv:2504.08525)  
Task Memory Tree for multi-step agent tasks: step I/O, sub-task relations, task-relationship inference, prompt synthesis; designed for future graph-aware (DAG) extensions. **Difference from ours:** We emphasize user utterance as trigger (Turn → UserUtterance → Steps) and token efficiency (summaries + refs); we also model life-long topic/preference/people layer.

**AriGraph** (knowledge graph + episodic memory)  
Unified memory graph (semantic + episodic) that agents build and update in interactive environments; improves over flat memory and RAG on complex reasoning. **Difference from ours:** Graph-oriented, environment-focused; we use a tree schema with life-long and task layers and path-based retrieval for conversational helping.

**Optimus-1** (NeurIPS 2024)  
Hybrid multimodal memory: hierarchical knowledge graph + abstracted experience pool, with planner and reflector modules. **Difference from ours:** Multimodal and environment-oriented; we focus on conversational life-long helping with user-triggered task structure and ref-based step storage.

---

## 4. RL for memory and retrieval

**Search-R1** (Jin et al., 2025)  
Trains LLMs with RL (GRPO) to decide when and how to use search; extends to search-augmented agents. **Difference from ours:** We apply the same RL-for-access idea to **memory structure** (ADD/UPDATE/DELETE on a tree) and to a two-layer schema.

**Memory-R1** (Yan et al., 2025)  
Two-agent pipeline: Memory Manager (ADD/UPDATE/DELETE/NOOP on flat memory) and Answer Agent (retrieval + distillation + answer), both trained with GRPO. **Difference from ours:** We use a **tree** (life-long + task) and **user-triggered** turns (Turn, UserUtterance, Step with refs); same RL framing, richer structure.

---

## 5. Generative agents and reflection

**Generative Agents** (Park et al., 2023)  
Agents with memory stream, reflection (synthesizing memories into higher-level insights), and retrieval for planning. **Difference from ours:** Stream + reflection, no formal two-layer tree; we have life-long + task schema and Steps/Outcomes instead of a separate reflection stream.

**"My agent understands me better"** (arXiv:2404.00573)  
Dynamic human-like memory consolidation (relevance, time, recall frequency) for LLM agents. **Difference from ours:** We model consolidation via ADD/UPDATE and topic-level structure; we also formalize task-level trace (turns, steps, refs).

---

## 6. Life-long and preference memory

**Learning User Preferences Through Interaction** (arXiv:2601.02702)  
Multi-session collaboration benchmark: agents learn user preferences over time; persistent memory improves task success and efficiency. **Difference from ours:** We give a **structure** for preferences (Preferences/People nodes under Topics) and for task-level execution (turns, steps).

**Enabling Personalized Long-term Interactions** (arXiv:2510.07925)  
Persistent memory and user profiles for long-term personalized LLM agents. **Difference from ours:** We formalize where preferences and people sit in the tree (per-topic/per-project) and how task execution (turns, steps, refs) is stored and retrieved.
