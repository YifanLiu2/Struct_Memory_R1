# Datasets for Evaluation

Table of evaluation datasets: domain, scale, what they measure, and recommendation (primary / secondary / optional). See [references.bib](references.bib) for citations.

| Dataset | Domain | Scale | What it measures | Recommendation |
|---------|--------|--------|-------------------|----------------|
| **LoCoMo** | Multi-session dialogue | ~600 turns, ~26k tokens per convo; 10 conversations; QA split 1:1:8 | Long-term QA; single-hop, multi-hop, temporal, open-domain | **Primary** (already in use) |
| **LongMemEval** | Multi-session dialogue | 500 questions; S/M/oracle versions (~40–500 sessions) | Information extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention | **Primary** for long-term memory |
| **Semantic XPath domains** | Itinerary, To-Do, Meal Kit | 20 QA each; structured trees (Itinerary→Day→POI, etc.) | Path/semantic retrieval over domain trees | **Primary** for OOD structured access |
| **ConvoMem** | General conversational | 75k+ QA pairs; categories: user facts, assistant recall, abstention, preferences, temporal, implicit | When RAG helps; user facts, preferences, temporal change | **Secondary** (scale; ablation RAG vs full context) |
| **EvolMem** | Multi-session, cognitive | Controllable multi-session conversations | Declarative / non-declarative memory, fine-grained abilities | **Optional** if adding cognitive angles |
| **Conversation Chronicles** | Multi-session | 1M dialogues | Temporal dynamics, speaker relationships | **Optional** for scale/temporal analysis |
| **GTA** | Tool-use, multi-step | 229 real-world tasks; multimodal | General tool agents; real tools, real queries | **Optional** if evaluating task-level (tool) steps |
| **MCPVerse** | Agentic tool use | 550+ tools, 147k+ action space | Outcome-based; large tool sets | **Optional** for agentic/tool-use evaluation |
| **M3-Bench** | Multimodal tool use | 28 servers, 231 tools; multi-hop, multi-threaded | MCP tool use, workflow consistency | **Optional** for multi-step agent evaluation |

## Recommendations

- **Primary:** LoCoMo, LongMemEval, Semantic XPath domains (Itinerary, To-Do, Meal Kit). Cover long-term QA, multi-session reasoning, and OOD structured retrieval.
- **Secondary:** ConvoMem (when RAG helps; compare with Mem0 or full-context on same setup).
- **Optional:** EvolMem (cognitive memory), Conversation Chronicles (scale/temporal), or tool-use benchmarks (GTA, MCPVerse, M3-Bench) if evaluating task-level step/tool retrieval and agentic behavior.
