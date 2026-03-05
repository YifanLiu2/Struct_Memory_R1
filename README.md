# Structured Memory-R1

**RL-Based Memory Management for Structured Memory in LLM Agents**

Yifan Liu, Liam Gallagher, David Courtis, Jiazhou Liang  
University of Toronto &middot; MIE1630 Winter 2026

> **Report:** [`struct_memory_R1_latex/implementation.pdf`](struct_memory_R1_latex/implementation.pdf) &middot; **Proposal:** [`struct_memory_R1_latex/main_neurips.tex`](struct_memory_R1_latex/main_neurips.tex)

---

## 1 &ensp; Motivation

LLM agents increasingly maintain external memory across interactions, yet most systems treat memory access as a static, heuristic process.  
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) showed that reinforcement learning (GRPO) can teach an LLM *when and how* to issue search queries; [Memory-R1](https://arxiv.org/abs/2508.19828) applied the same idea to flat memory banks.  
**Structured Memory-R1** takes the next step: we train agents with RL to navigate *tree-structured* memory, where nodes are organized in compositional hierarchies (e.g. `Itinerary -> Day -> POI`) rather than treated as an unordered bag of entries.

The core loop is identical to Search-R1, with one substitution:

```
User question
    |
    v
LLM reasons in <think>...</think>
    |
    v
LLM emits  <memory> natural-language query </memory>      (was <search>)
    |
    v
Memory server returns relevant entries / subtrees
    |
    v
Results injected as  <information>...</information>
    |
    v
LLM continues reasoning  -- or --  emits <answer>...</answer>
```

The policy is optimized with **Group Relative Policy Optimization (GRPO)**: for each question we sample K trajectories, score them with exact-match reward, and update the policy using group-normalized advantages. Tokens inside `<information>` blocks are masked from the policy loss.

---

## 2 &ensp; Methods Compared

| Method | RL? | Memory Type | Description |
|---|---|---|---|
| **In-Context Memory** | No | Flat | Entire memory serialized into the LLM prompt |
| **Semantic XPath** | No | Structured | LLM generates XPath-style queries executed over the tree ([paper](https://github.com/D3Mlab/SemanticXpath-Chat)) |
| **Memory-R1 (Flat)** | GRPO | Flat | RL-trained agent queries a flat memory bank |
| **Struct Memory-R1** | GRPO | Structured | RL-trained agent queries tree-structured memory |

In-Context Memory and Semantic XPath are implemented as non-RL baselines and discussed in the report. The codebase implements the two RL methods.

---

## 3 &ensp; Repository Structure

```
.
├── memory_r1/                          # === Our implementation ===
│   ├── memory_tree.py                  #   MemoryNode / MemoryTree (rooted tree data model)
│   ├── flat_memory.py                  #   FlatMemoryStore (list + cosine similarity)
│   ├── memory_server.py                #   FastAPI retrieval server (POST /retrieve)
│   ├── llm_agent/
│   │   └── generation.py               #   MemoryLLMGenerationManager (multi-turn agent loop)
│   └── data/
│       ├── itinerary.json              #   Travel itinerary domain  (20 QA pairs)
│       ├── todo.json                   #   To-do list domain        (20 QA pairs)
│       ├── mealkit.json                #   Meal-kit domain          (20 QA pairs)
│       └── locomo_converter.py         #   LoCoMo -> structured tree converter
│
├── scripts/data_process/
│   └── memory_data.py                  #   QA data -> parquet (training-ready format)
│
├── train_memory_grpo.sh                #   GRPO training launcher
│
├── verl/                               # === veRL RL framework (upstream dependency) ===
│   ├── trainer/
│   │   ├── main_memory.py              #   * Training entry point (MemoryRewardManager)
│   │   ├── main_ppo.py                 #   Original Search-R1 entry point
│   │   ├── ppo/ray_trainer.py          #   Core GRPO / PPO loop (Ray-based)
│   │   └── config/ppo_trainer.yaml     #   Default Hydra config
│   └── utils/reward_score/
│       └── qa_em.py                    #   Exact-match / sub-EM scoring
│
├── search_r1/                          # === Search-R1 base framework (upstream dependency) ===
│   ├── llm_agent/
│   │   ├── generation.py               #   LLMGenerationManager (original search version)
│   │   └── tensor_helper.py            #   Tensor padding / masking utilities (shared)
│   └── search/
│       └── retrieval_server.py         #   Original corpus retrieval server
│
├── struct_memory_R1_latex/             # === Report & proposal ===
│   ├── implementation.tex / .pdf       #   Intermediate implementation report
│   ├── main_neurips.tex                #   Project proposal
│   ├── reference.bib                   #   Bibliography
│   └── neurips_2023.sty                #   NeurIPS style
│
├── requirements.txt                    # Python dependencies
├── setup.py / pyproject.toml           # Package installation
└── LICENSE                             # Apache 2.0
```

Files marked with **\*** are new; everything under `verl/` and `search_r1/` is reused from Search-R1 unless noted.

---

## 4 &ensp; Key Components

### 4.1 &ensp; Memory Data Model (`memory_r1/memory_tree.py`)

Memory is represented as a rooted tree **M = (V, E, r)**:

- **`MemoryNode`** &mdash; dataclass with `node_id`, `node_type`, `attributes` (dict of text), `children`, `parent`.
- **`MemoryTree`** &mdash; wraps the root node with an ID index and provides:
  - `keyword_search(query, topk)` &mdash; token-overlap scoring over all nodes
  - `subtree_search(query, topk)` &mdash; returns formatted subtrees for top matches
  - `semantic_navigate(query)` &mdash; type-aware navigation (parses node types from the query)
  - `to_flat_entries()` &mdash; flatten the tree for the flat-memory baseline
  - `to_json()` / `from_json()` &mdash; JSON serialization

### 4.2 &ensp; Flat Memory Store (`memory_r1/flat_memory.py`)

- **`FlatMemoryStore`** &mdash; list of `MemoryEntry` objects with keyword or embedding-based top-k retrieval.
- Accepts an optional `embedding_fn` for dense retrieval; falls back to keyword overlap when none is provided.

### 4.3 &ensp; Memory Retrieval Server (`memory_r1/memory_server.py`)

FastAPI server exposing `POST /retrieve` with the same request schema as Search-R1's retrieval server:

```json
{
  "queries": ["conference sessions on day 2"],
  "topk": 3,
  "return_scores": true,
  "memory_type": "structured"
}
```

Supports both `"flat"` and `"structured"` memory types. Can be loaded with any domain JSON at startup.

### 4.4 &ensp; LLM Agent Loop (`memory_r1/llm_agent/generation.py`)

`MemoryLLMGenerationManager` adapts Search-R1's `LLMGenerationManager`:

| What changed | Detail |
|---|---|
| Action tags | `<search>` / `</search>` &rarr; `<memory>` / `</memory>` |
| Environment step | `batch_search()` &rarr; `batch_memory_retrieve()` (calls the memory server) |
| Result formatting | `_passages2string()` &rarr; `_memory_results_to_string()` (includes tree paths for structured mode) |
| Config | `GenerationConfig` &rarr; `MemoryGenerationConfig` (adds `memory_url`, `memory_type`) |

Everything else&mdash;the multi-turn `run_llm_loop`, rolling-state updates, info masking, GPU-padding wrapper, final output composition&mdash;is identical to Search-R1.

### 4.5 &ensp; Reward & Training (`verl/trainer/main_memory.py`)

- **`MemoryRewardManager`** &mdash; scores trajectories using:
  - *Substring exact match* for QA datasets (`locomo`, `locomo_structured`)
  - *Constraint pass rate* for structured domains (`itinerary`, `todo`, `mealkit`)
  - A small format bonus (+0.1) when the agent correctly uses `<memory>` tags
- Training is launched with `train_memory_grpo.sh`, which calls `verl.trainer.main_memory` under GRPO with state masking on `<information>` blocks.

---

## 5 &ensp; Evaluation Data

### Semantic XPath Domains (curated)

Three domains from the [Semantic XPath](https://github.com/D3Mlab/SemanticXpath-Chat) evaluation setup, each stored as a JSON file containing the memory tree and 20 QA pairs:

| Domain | Schema | Example question |
|---|---|---|
| Travel Itinerary | `Itinerary -> Version -> Day -> POI` | *Which day is packed with conference sessions?* |
| To-Do List | `TodoList -> Category -> Project -> Task` | *What high-priority tasks are still pending?* |
| Meal Kit | `MealPlan -> Day -> Meal -> Option` | *Which breakfast on Monday is vegetarian and gluten-free?* |

### Structured LoCoMo (converted)

The [LoCoMo](https://arxiv.org/abs/2402.17753) multi-session dialogue benchmark, converted into structured trees:

```
Dialogue -> Session -> Topic -> MemoryEntry
```

The converter (`memory_r1/data/locomo_converter.py`) also produces a flat version for the Memory-R1 baseline. LoCoMo's existing QA pairs are reused for evaluation.

---

## 6 &ensp; Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPUs (8x recommended for GRPO training)
- `pip install -e . && pip install -r requirements.txt`

### Step 1 &mdash; Prepare training data

```bash
python scripts/data_process/memory_data.py \
    --data_dir memory_r1/data \
    --output_dir data/memory_train
```

This reads the domain JSONs, applies the memory prompt template, and writes `train.parquet` / `test.parquet`.

### Step 2 &mdash; Launch the memory server

In a **separate terminal**:

```bash
# Structured memory (e.g. itinerary domain)
python -m memory_r1.memory_server \
    --structured_memory_path memory_r1/data/itinerary.json \
    --port 8000

# Flat memory (e.g. LoCoMo)
python -m memory_r1.memory_server \
    --flat_memory_path memory_r1/data/locomo/locomo_flat.json \
    --port 8000
```

### Step 3 &mdash; Train with GRPO

```bash
bash train_memory_grpo.sh
```

Key config knobs (edit the script or pass as overrides):

| Parameter | Default | Meaning |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-3B` | HuggingFace model ID |
| `MEMORY_TYPE` | `structured` | `flat` or `structured` |
| `max_turns` | `3` | Max memory-query rounds per trajectory |
| `retriever.topk` | `5` | Entries returned per query |
| `actor_rollout_ref.rollout.n_agent` | `5` | GRPO group size K |
| `trainer.total_training_steps` | `500` | Total GRPO steps |

---

## 7 &ensp; Reproducibility

| Mechanism | Detail |
|---|---|
| **Fixed seed** | `random.seed(42)` in data processing |
| **Deterministic data** | All 60 QA pairs and memory trees are checked into `memory_r1/data/` |
| **Pinned config** | Training hyperparameters are explicit in `train_memory_grpo.sh` |
| **State masking** | `<information>` blocks are masked from the policy gradient (same as Search-R1) |
| **GRPO** | Group-relative advantages remove the need for absolute reward calibration |

---

## 8 &ensp; Acknowledgments

This project builds on the following open-source work:

- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) (Jin et al., 2025) &mdash; RL for search-augmented LLMs
- [Memory-R1](https://arxiv.org/abs/2508.19828) (Yan et al., 2025) &mdash; RL for flat memory management
- [Semantic XPath](https://github.com/D3Mlab/SemanticXpath-Chat) (Liu et al., 2026) &mdash; Structured memory access for ConvAI
- [veRL / HybridFlow](https://github.com/volcengine/verl) (Sheng et al., 2025) &mdash; RL training framework for LLMs
- [LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., 2024) &mdash; Long-term conversational memory benchmark

## License

Apache 2.0 (inherited from Search-R1 / veRL).
