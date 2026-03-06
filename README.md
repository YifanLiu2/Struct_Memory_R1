# Structured Memory-R1

**RL-Based Memory Management for Structured Memory in LLM Agents**

Yifan Liu, Liam Gallagher, David Courtis, Jiazhou Liang  
University of Toronto &middot; MIE1630 Winter 2026

> **Report:** [`struct_memory_R1_latex/implementation.pdf`](struct_memory_R1_latex/implementation.pdf) &middot; **Proposal:** [`struct_memory_R1_latex/main_neurips.tex`](struct_memory_R1_latex/main_neurips.tex)

---

## 1 &ensp; Motivation

LLM agents increasingly maintain external memory across interactions, yet most systems treat memory access as a static, heuristic process.  
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) showed that RL (GRPO) can teach an LLM *when and how* to issue search queries. [Memory-R1](https://arxiv.org/abs/2508.19828) extended this to full memory management &mdash; learning ADD, UPDATE, DELETE, and NOOP operations on a flat memory bank via two specialized RL agents.

**Structured Memory-R1** takes the next step: we apply the same two-agent RL framework to *tree-structured* memory, where nodes are organized in compositional hierarchies (e.g. `Dialogue -> Session -> Topic -> MemoryEntry`) rather than treated as an unordered bag of entries. The Memory Manager must now make structural decisions &mdash; not just *what* to store but *where in the tree* to place it.

---

## 2 &ensp; Architecture (Two-Agent Pipeline)

Faithful to the Memory-R1 paper, our system has **two independently-trained agents**:

```
Stage 1: Memory Bank Construction
  Dialogue Turn -> LLM Fact Extraction -> Retrieve Related Memories
      -> Memory Manager decides {ADD, UPDATE, DELETE, NOOP}
      -> Memory Bank is updated

Stage 2: Answer Generation
  Question -> Retrieve Top-60 Memories from Bank
      -> Answer Agent applies Memory Distillation (selects relevant subset)
      -> Generates concise answer
```

**Training:** The Memory Manager is trained first (Answer Agent frozen, provides EM reward). Then the Answer Agent is trained on memory banks produced by the trained Memory Manager. Both use GRPO.

---

## 3 &ensp; Methods Compared

| Method | RL? | Memory Type | Memory Ops | Description |
|---|---|---|---|---|
| **In-Context Memory** | No | Flat | Read | Full memory serialized into prompt |
| **Semantic XPath** | No | Structured | Read | LLM generates XPath queries over tree |
| **Memory-R1 (Flat)** | GRPO | Flat | ADD/UPDATE/DELETE/NOOP + Read | Two-agent pipeline on flat bank |
| **Struct Memory-R1** | GRPO | Structured | ADD/UPDATE/DELETE/NOOP + Read | Two-agent pipeline on tree memory |

---

## 4 &ensp; Repository Structure

```
.
├── memory_r1/                              # === Our implementation ===
│   ├── memory_tree.py                      #   MemoryNode / MemoryTree data model
│   ├── flat_memory.py                      #   FlatMemoryStore with CRUD operations
│   ├── memory_server.py                    #   FastAPI retrieval server
│   │
│   ├── memory_manager/                     #   Stage 1: Memory Manager Agent
│   │   ├── flat_manager.py                 #     Flat ops: parse JSON, apply ADD/UPDATE/DELETE/NOOP
│   │   ├── tree_manager.py                 #     Tree ops: parent_id for ADD, subtree DELETE
│   │   ├── generation.py                   #     One-shot GRPO generation loop
│   │   └── prompts.py                      #     Prompt templates (paper Figs 9-10 + tree ext)
│   │
│   ├── answer_agent/                       #   Stage 2: Answer Agent
│   │   ├── answer_agent.py                 #     Memory Distillation + answer extraction
│   │   ├── generation.py                   #     One-shot GRPO generation loop
│   │   └── prompts.py                      #     Prompt templates (paper Fig 11 + tree ext)
│   │
│   ├── evaluation.py                       #   F1, BLEU-1, EM, LLM-as-a-Judge
│   ├── inference.py                        #   End-to-end pipeline (Stage 1 + Stage 2)
│   │
│   ├── llm_agent/
│   │   └── generation.py                   #   Multi-turn agent loop (Search-R1 adaptation)
│   │
│   └── data/
│       ├── data_construction.py            #   GPT-based fact extraction + training data builder
│       ├── itinerary.json                  #   Travel itinerary domain (20 QA pairs)
│       ├── todo.json                       #   To-do list domain (20 QA pairs)
│       ├── mealkit.json                    #   Meal-kit domain (20 QA pairs)
│       └── locomo_converter.py             #   LoCoMo -> structured tree converter
│
├── verl/                                   # === veRL RL framework ===
│   ├── trainer/
│   │   ├── main_memory_manager.py          #   * Memory Manager GRPO training entry
│   │   ├── main_answer_agent.py            #   * Answer Agent GRPO training entry
│   │   ├── main_memory.py                  #   Single-agent training (intermediate version)
│   │   ├── main_ppo.py                     #   Original Search-R1 entry point
│   │   └── ppo/ray_trainer.py              #   Core GRPO/PPO loop (Ray-based)
│   └── utils/reward_score/
│       └── qa_em.py                        #   Exact-match / sub-EM scoring
│
├── search_r1/                              # === Search-R1 base framework ===
│   ├── llm_agent/
│   │   ├── generation.py                   #   Original search agent loop
│   │   └── tensor_helper.py                #   Tensor utilities (shared)
│   └── search/                             #   Retrieval server implementations
│
├── scripts/data_process/
│   └── memory_data.py                      #   Legacy data processing
│
├── train_memory_manager.sh                 #   Stage 1 training launcher
├── train_answer_agent.sh                   #   Stage 2 training launcher
├── train_memory_grpo.sh                    #   Single-agent training (intermediate)
│
├── struct_memory_R1_latex/                 #   Report & proposal
│   ├── implementation.tex / .pdf
│   ├── main_neurips.tex
│   └── reference.bib
│
├── requirements.txt
├── setup.py / pyproject.toml
└── LICENSE
```

---

## 5 &ensp; Key Components

### 5.1 &ensp; Memory Manager (`memory_r1/memory_manager/`)

The Memory Manager processes each dialogue turn and decides how to update the memory bank. For each extracted fact, it outputs one of:

- **ADD** &mdash; insert a new entry (flat: append to list; tree: specify `parent_id`)
- **UPDATE** &mdash; merge new information into an existing entry
- **DELETE** &mdash; remove an outdated or contradictory entry
- **NOOP** &mdash; no change needed

Output format is JSON:
```json
{"memory": [
  {"id": "0", "text": "User likes pizza", "event": "NONE"},
  {"id": "1", "text": "Adopted 2 dogs: Buddy and Scout", "event": "UPDATE",
   "old_memory": "Adopted a dog named Buddy"}
]}
```

For Struct Memory-R1, ADD operations include `parent_id` and `node_type` to specify where in the tree to insert.

**Reward:** The Memory Manager's operations are judged by whether the resulting memory bank enables the frozen Answer Agent to answer correctly (EM score, paper Eq. 4).

### 5.2 &ensp; Answer Agent (`memory_r1/answer_agent/`)

Given a question and top-60 retrieved memories, the Answer Agent:
1. **Memory Distillation** &mdash; selects the most relevant memories from the retrieved set
2. **Answer generation** &mdash; produces a concise answer (< 5-6 words)

This two-step process filters noise from retrieval, avoiding the "lost in the middle" problem. The agent outputs selected memories first, then the answer after `**Answer:**`.

**Reward:** Exact Match between predicted and gold answer (paper Eq. 4).

### 5.3 &ensp; Data Construction (`memory_r1/data/data_construction.py`)

Following the paper's Algorithms 1-2:
- Uses GPT-4o-mini to extract facts from LoCoMo dialogue turns
- Builds temporal memory banks from preceding ~50 turns
- Creates separate parquet files for Memory Manager and Answer Agent training
- Also processes Semantic XPath domains as additional evaluation data

### 5.4 &ensp; Evaluation (`memory_r1/evaluation.py`)

Three metrics from the paper:
- **F1** &mdash; token-level F1 between prediction and gold answer
- **BLEU-1** &mdash; unigram BLEU score
- **LLM-as-a-Judge** &mdash; GPT labels each answer as CORRECT/WRONG

Supports evaluation by question type (single-hop, multi-hop, open-domain, temporal).

---

## 6 &ensp; Evaluation Data

### LoCoMo (primary benchmark)

The [LoCoMo](https://arxiv.org/abs/2402.17753) benchmark: long multi-session dialogues (~600 turns, 26k tokens) with QA pairs covering single-hop, multi-hop, open-domain, and temporal reasoning. Following Memory-R1, we use a 1:1:8 train/val/test split (152/81/1307 questions).

### Semantic XPath Domains

Three domains from [Semantic XPath](https://github.com/D3Mlab/SemanticXpath-Chat), each with a structured memory tree and 20 QA pairs:

| Domain | Schema | Example question |
|---|---|---|
| Travel Itinerary | `Itinerary -> Version -> Day -> POI` | *Which day is packed with conference sessions?* |
| To-Do List | `TodoList -> Category -> Project -> Task` | *What high-priority tasks are still pending?* |
| Meal Kit | `MealPlan -> Day -> Meal -> Option` | *Which breakfast on Monday is vegetarian?* |

---

## 7 &ensp; Quick Start

### Prerequisites

- Python 3.10+, CUDA-capable GPUs (8x recommended)
- `pip install -e . && pip install -r requirements.txt`
- `OPENAI_API_KEY` for data construction and LLM-as-a-Judge evaluation

### Step 1 &mdash; Prepare training data

```bash
python -m memory_r1.data.data_construction \
    --locomo_path data/locomo/locomo.json \
    --data_dir memory_r1/data \
    --output_dir data/memory_r1_train \
    --use_gpt
```

### Step 2 &mdash; Train Memory Manager (Stage 1)

```bash
bash train_memory_manager.sh
```

### Step 3 &mdash; Train Answer Agent (Stage 2)

After Stage 1 completes, rebuild memory banks with the trained MM, then:

```bash
bash train_answer_agent.sh
```

### Step 4 &mdash; Run inference

```bash
python -m memory_r1.inference \
    --locomo_path data/locomo/locomo.json \
    --memory_type flat \
    --model_path openai \
    --output_path results.json
```

### Step 5 &mdash; Evaluate

```bash
python -m memory_r1.inference \
    --mode evaluate \
    --output_path results.json \
    --use_judge
```

---

## 8 &ensp; Training Details

### Memory Manager

| Parameter | Value |
|---|---|
| RL Algorithm | GRPO |
| Reward | EM from frozen Answer Agent |
| Actions | ADD, UPDATE, DELETE, NOOP |
| Group size (K) | 5 |
| Learning rate | 1e-6 |
| Max response length | 1024 tokens |

### Answer Agent

| Parameter | Value |
|---|---|
| RL Algorithm | GRPO |
| Reward | EM(predicted, gold) |
| Retrieved memories | Top-60 per question |
| Group size (K) | 5 |
| Learning rate | 1e-6 |
| Max response length | 2048 tokens |

---

## 9 &ensp; Reproducibility

| Mechanism | Detail |
|---|---|
| **Fixed seed** | `random.seed(42)` in data processing |
| **Deterministic data** | QA pairs and memory trees are checked into `memory_r1/data/` |
| **Pinned config** | All hyperparameters are explicit in training scripts |
| **Separate training** | MM and AA are trained independently for attribution clarity |
| **GRPO** | Group-relative advantages remove the need for absolute reward calibration |

---

## 10 &ensp; Acknowledgments

- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) (Jin et al., 2025) &mdash; RL for search-augmented LLMs
- [Memory-R1](https://arxiv.org/abs/2508.19828) (Yan et al., 2025) &mdash; RL for memory management (our primary reference)
- [Semantic XPath](https://github.com/D3Mlab/SemanticXpath-Chat) (Liu et al., 2026) &mdash; Structured memory access for ConvAI
- [veRL / HybridFlow](https://github.com/volcengine/verl) (Sheng et al., 2025) &mdash; RL training framework for LLMs
- [LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., 2024) &mdash; Long-term conversational memory benchmark

## License

Apache 2.0 (inherited from Search-R1 / veRL).
