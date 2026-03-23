# Semantic XPath: Hierarchical Retrieval with Score Fusion

A framework for querying and modifying hierarchical data using natural language with **semantic predicates**, **hierarchical aggregation**, **score fusion**, **CRUD operations**, and **in-tree versioning**.

## Core Idea

Traditional XPath operates on exact matches. Semantic XPath extends this with:
- **Semantic matching**: Match by meaning, not just text
- **Probabilistic scores**: Every match has a confidence score in [0, 1]
- **Hierarchical aggregation**: Aggregate scores from children to parents
- **Score fusion**: Combine scores across query steps (product)
- **CRUD Operations**: Create, Read, Update, Delete nodes using natural language
- **In-Tree Versioning**: Every modification creates a new version within the tree

```
User: "find museums in artistic days"

Semantic XPath: Read(/Itinerary/Version[-1]/Day[agg_prev(POI[atom(content =~ "artistic")])]/POI[atom(content =~ "museum")])

Result: [Art Gallery of Ontario: 0.95, Royal Ontario Museum: 0.89, ...]
```

```
User: "delete all the museums"

Semantic XPath: Delete(/Itinerary/Version[-1]/Day/POI[atom(content =~ "museum")])

Result: Deleted 2 nodes, created Version 2
```

```
User: "in the version where I deleted museums, update the first POI to chinese food"

Semantic XPath: Update(/Itinerary/Version[atom(content =~ "delete museum")]/Day/POI[1], type: chinese food)

Result: Found Version 2 via semantic matching, updated POI, created Version 3
```

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [End-to-End Flow](#end-to-end-flow)
4. [In-Tree Versioning](#in-tree-versioning)
5. [CRUD Operations](#crud-operations)
6. [REST API](#rest-api)
7. [Mathematical Foundation](#mathematical-foundation)
8. [Query Syntax](#query-syntax)
9. [Predicate Types](#predicate-types)
10. [Schema System](#schema-system)
11. [Usage](#usage)
12. [Configuration](#configuration)
13. [Project Structure](#project-structure)
14. [Quick Reference](#quick-reference)

---

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (for LLM scoring and query generation)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/LLM-VM.git
cd LLM-VM

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your OpenAI API key in `config.yaml`:

```yaml
openai:
  api_key: "your-api-key-here"  # Or use ${OPENAI_API_KEY} for env var
  model: "gpt-4o"
```

---

## Quick Start

### Interactive CLI

```bash
# Start the interactive pipeline
python -m pipeline.semantic_xpath_pipeline

# With custom options
python -m pipeline.semantic_xpath_pipeline --scoring entailment --top-k 10
```

### REST API Server

```bash
# Start the API server
python -m api.run --port 5000

# With options
python -m api.run --port 5000 --scoring entailment --debug
```

### Python API

```python
from pipeline import SemanticXPathPipeline

pipeline = SemanticXPathPipeline()

# Execute natural language queries
result = pipeline.process_request("find museums in the itinerary")
result = pipeline.process_request("delete all the museums")
result = pipeline.process_request("add a sushi restaurant on day 1")

print(pipeline.format_result(result))
```

---

## End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SEMANTIC XPATH PIPELINE                            │
└──────────────────────────────────────────────────────────────────────────────┘

     User Request                    LLM Query                   Structured
    (Natural Language)              Generator                      Query
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  "find museums  │           │   GPT-4 with    │           │   /Itinerary/   │
│   in artistic   │  ──────▶  │  schema-aware   │  ──────▶  │   Day[mass(...)]│
│     days"       │           │    prompts      │           │   /POI[sem(...)]│
└─────────────────┘           └─────────────────┘           └─────────────────┘
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              QUERY EXECUTION                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. PARSE QUERY                                                             │
│      ┌─────────────────────────────────────────────────────┐                │
│      │ Query: /Itinerary/Day[agg_prev(POI[atom(...)])]/POI[atom()]          │
│      │                                                                       │
│      │ Steps:                                                                │
│      │   Step 0: Itinerary (root)                                           │
│      │   Step 1: Day [predicate: agg_prev(POI[atom(content =~ "artistic")])]│
│      │   Step 2: POI [predicate: atom(content =~ "museum")]                 │
│      └─────────────────────────────────────────────────────┘                │
│                                                                              │
│   2. BFS TRAVERSAL WITH SCORING                                             │
│      ┌─────────────────────────────────────────────────────┐                │
│      │ Step 0: [Itinerary] ─── root                                         │
│      │            │                                                          │
│      │ Step 1: [Day1, Day2, Day3] ─── expand to children                    │
│      │            │                                                          │
│      │         Score each Day with agg_prev(POI[atom("artistic")])          │
│      │            │  Day1: 0.72 (average over POI scores)                   │
│      │            │  Day2: 0.85                                              │
│      │            │  Day3: 0.45                                              │
│      │            │                                                          │
│      │ Step 2: [POI1, POI2, ...] ─── expand to children                     │
│      │            │                                                          │
│      │         Score each POI with atom("museum")                           │
│      │            │  Art Gallery: 0.92 (local score)                        │
│      │            │  CN Tower: 0.08                                          │
│      │            │  Royal Ontario Museum: 0.95                             │
│      └─────────────────────────────────────────────────────┘                │
│                                                                              │
│   3. SCORE FUSION (PRODUCT)                                                  │
│      ┌─────────────────────────────────────────────────────┐                │
│      │ For each final node, multiply scores from all steps:                 │
│      │                                                                       │
│      │ Art Gallery of Ontario (in Day 1):                                   │
│      │   Step 1: Day "artistic" score = 0.72                                │
│      │   Step 2: POI "museum" score = 0.92                                  │
│      │   Final score = 0.72 × 0.92 = 0.662                                  │
│      │                                                                       │
│      │                                                                       │
│      │                                                                       │
│      └─────────────────────────────────────────────────────┘                │
│                                                                              │
│   4. FILTER & RANK                                                           │
│      ┌─────────────────────────────────────────────────────┐                │
│      │ Apply threshold (e.g., 0.5) and top_k (e.g., 5)                      │
│      │ Sort by final score descending                                        │
│      │                                                                       │
│      │ Results:                                                              │
│      │   1. Royal Ontario Museum: 0.98                                      │
│      │   2. Art Gallery of Ontario: 0.97                                    │
│      │   3. Casa Loma: 0.61                                                 │
│      └─────────────────────────────────────────────────────┘                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Role | Key Files |
|-----------|------|-----------|
| **Query Generator** | Unified NL → CRUD + XPath (single LLM call) | `xpath_query_generation/` |
| **Version Manager** | In-tree versioning, semantic version resolution | `tree_modification/version_manager.py` |
| **Parser** | Query string → AST (atom/agg_exists/agg_prev) | `dense_xpath/parser.py` |
| **Executor** | BFS traversal + fusion | `dense_xpath/dense_xpath_executor.py` |
| **Predicate Handler** | Scoring + aggregation | `dense_xpath/predicate_handler.py` |
| **Scorer** | Semantic similarity (also for version resolution) | `predicate_classifier/` |
| **Semantic XPath Orchestrator** | CRUD orchestration with versioning | `pipeline_execution/semantic_xpath_orchestrator.py` |

---

## In-Tree Versioning

The system uses **in-tree versioning** where each modification creates a new Version node within the tree structure itself.

### Version Structure

```xml
<Itinerary>
  <Version number="1">
    <patch_info></patch_info>
    <conversation_history></conversation_history>
    <Day index="1">...</Day>
    <Day index="2">...</Day>
  </Version>
  <Version number="2">
    <patch_info>Deleted: Royal Ontario Museum, Art Gallery of Ontario</patch_info>
    <conversation_history>delete the museum</conversation_history>
    <Day index="1">...</Day>
    <Day index="2">...</Day>
  </Version>
</Itinerary>
```

### Version Selectors

| Selector | Meaning | Example |
|----------|---------|---------|
| `Version[-1]` | Latest version (default) | `Read(/Itinerary/Version[-1]/Day/POI)` |
| `Version[N]` | Specific version number | `Read(/Itinerary/Version[2]/Day)` |
| `Version[atom(content =~ "...")]` | Semantic search on version metadata | `Read(/Itinerary/Version[atom(content =~ "delete museum")]/Day)` |

### Semantic Version Resolution

When using semantic version selectors, the system:
1. Builds descriptions from `patch_info` + `conversation_history` for each version
2. Scores all versions in a **single batch** using entailment scoring
3. Selects the **top-1** version by score (threshold > 0.5)

### Compound Version + Operation Queries

You can reference versions by their changes AND perform operations in one query:

```
User: "in the version that deleted the museum, update the first POI to chinese food"

Generated: Update(/Itinerary/Version[atom(content =~ "delete museum")]/Day/POI[1], type: chinese food)

Flow:
1. Semantic search finds Version 2 (deleted museums)
2. Update applied to Version 2's content
3. New Version 3 created with the changes
```

---

## CRUD Operations

The system supports full CRUD (Create, Read, Update, Delete) operations on tree data using natural language with a **unified LLM call** for both intent classification and query generation.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CRUD PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────┘

    User Query             Unified LLM Call           CRUD + XPath
   (Natural Lang)      (Intent + Query Gen)             Query
        │                       │                         │
        ▼                       ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌─────────────────────┐
│ "in version   │       │  Single LLM   │       │ Update(/Itinerary/  │
│  that deleted │  ──▶  │  generates    │  ──▶  │ Version[atom(...)]/ │
│  museum, ..." │       │  full query   │       │ Day/POI[1], ...)    │
└───────────────┘       └───────────────┘       └─────────────────────┘
                                                         │
                        ┌────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      VERSION RESOLUTION                                  │
│  Extract version selector → Semantic scoring (batched) → Top-1 select   │
└─────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC XPATH EXECUTION                            │
│  Find candidate nodes with semantic scoring                              │
└─────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM NODE REASONING                                  │
│  Batched LLM calls to select truly relevant nodes from candidates        │
└─────────────────────────────────────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┬───────────────┬───────────────┐
          ▼                           ▼               ▼               ▼
     ┌─────────┐                ┌─────────┐     ┌─────────┐     ┌─────────┐
     │  READ   │                │ DELETE  │     │ UPDATE  │     │ CREATE  │
     │ Return  │                │ Remove  │     │ Modify  │     │ Insert  │
     │ Results │                │ Nodes   │     │ Content │     │ New Node│
     └─────────┘                └────┬────┘     └────┬────┘     └────┬────┘
                                     │               │               │
                                     └───────────────┴───────────────┘
                                                     │
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │     CREATE NEW VERSION        │
                                     │  Copy content → Apply changes │
                                     │  → Append Version node        │
                                     └───────────────────────────────┘
                                                     │
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │     SAVE TREE                 │
                                     │  result/demo/<filename>.xml   │
                                     │  (all versions in one file)   │
                                     └───────────────────────────────┘
```

### Operation Types

| Operation | Description | Example Query |
|-----------|-------------|---------------|
| **Read** | Find and retrieve nodes | "find museums in the itinerary" |
| **Create** | Add new nodes | "add a sushi restaurant after lunch on day 1" |
| **Update** | Modify existing nodes | "change the CN Tower visit to 2pm" |
| **Delete** | Remove nodes | "delete all the museums" |

### Full Query Format

Each operation displays a full query showing the operation type, version selector, and XPath:

```
Read(/Itinerary/Version[-1]/Day/POI[atom(content =~ "museum")])
Delete(/Itinerary/Version[-1]/Day[2]/POI[atom(content =~ "cafe")])
Update(/Itinerary/Version[-1]/Day/POI[atom(content =~ "CN Tower")], time_block: 2:00 PM)
Create(/Itinerary/Version[-1]/Day[1], Restaurant, sushi restaurant for lunch)

# Compound queries with semantic version selection
Update(/Itinerary/Version[atom(content =~ "delete museum")]/Day/POI[1], type: chinese food)
Read(/Itinerary/Version[atom(content =~ "add restaurant")]/Day/POI)
```

### Step Timing

Each operation displays detailed timing for performance analysis:

```
⏱️  Step Timing:
---------------------------------------------
  Query Generation             622.1ms  █░░░░░░░░░░░░░░░░░░░   7.9%
  Version Resolution           616.1ms  █░░░░░░░░░░░░░░░░░░░   7.8%
  Semantic XPath Execution       0.8ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
  LLM Node Reasoning          2468.9ms  ██████░░░░░░░░░░░░░░  31.3%
  Version Copy                   0.1ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
  LLM Content Update          4184.9ms  ██████████░░░░░░░░░░  53.0%
  Tree Modification              0.0ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
  Create New Version             1.6ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
---------------------------------------------
  TOTAL                       7894.5ms
```

### Output Files

Modified trees are saved to the `result/demo/` folder. The tree file contains all versions:

```
result/
└── demo/
    └── travel_memory_3day.xml    # Contains Version 1, 2, 3, ... all in one file

# Example tree structure with versions:
<Itinerary>
  <Version number="1">...</Version>
  <Version number="2">...</Version>
  <Version number="3">...</Version>
</Itinerary>
```

### CRUD Usage

```python
from pipeline import SemanticXPathPipeline

pipeline = SemanticXPathPipeline()

# Read operation (default: Version[-1])
result = pipeline.process_request("find museums in the itinerary")
# Displays: Read(/Itinerary/Version[-1]/Day/POI[atom(content =~ "museum")])

# Delete operation → creates new version
result = pipeline.process_request("delete all the museums")
# Displays: Delete(/Itinerary/Version[-1]/Day/POI[atom(content =~ "museum")])
# Creates: Version 2 in the tree

# Update operation → creates new version
result = pipeline.process_request("change the CN Tower visit to 2pm")
# Displays: Update(/Itinerary/Version[-1]/Day/POI[atom(content =~ "CN Tower")], time_block: 2:00 PM)
# Creates: Version 3 in the tree

# Compound query: reference version by changes + perform operation
result = pipeline.process_request("in the version that deleted museums, update the first POI to chinese food")
# Displays: Update(/Itinerary/Version[atom(content =~ "delete museum")]/Day/POI[1], type: chinese food)
# Semantic scoring finds Version 2, applies update, creates Version 4

# Create operation
result = pipeline.process_request("add a sushi restaurant after lunch on day 1")
# Displays: Create(/Itinerary/Version[-1]/Day[1], Restaurant, sushi restaurant after lunch)
# Creates: Version 5 in the tree

# View version history
history = pipeline.executor.version_manager.get_version_history(pipeline.executor.tree)
for v in history:
    print(f"Version {v['number']}: {v['patch_info']}")
```

---

## REST API

The framework includes a Flask-based REST API for integration with web applications.

### Starting the Server

```bash
# Basic usage
python -m api.run

# With options
python -m api.run --port 5000 --host 0.0.0.0 --scoring entailment --debug
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Execute a natural language CRUD query |
| `GET` | `/api/tree` | Get current tree state |
| `GET` | `/api/tree/versions` | List all tree versions |
| `GET` | `/api/tree/version/<id>` | Get specific version |
| `POST` | `/api/tree/reset` | Reset tree to original state |
| `GET` | `/api/config` | Get current configuration |
| `PUT` | `/api/config` | Update configuration |
| `GET` | `/api/health` | Health check |

### Execute Query

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "find museums in the itinerary"}'
```

**Response:**

```json
{
  "success": true,
  "operation": "READ",
  "full_query": "Read(/Itinerary/Version[-1]/Day/POI[atom(content =~ \"museum\")])",
  "selected_nodes": [
    {
      "type": "POI",
      "name": "Royal Ontario Museum",
      "tree_path": "Itinerary > Version 1 > Day 1 > Royal Ontario Museum",
      "description": "World-renowned museum featuring art, culture, and natural history..."
    }
  ],
  "timing": {
    "query_generation_ms": 622.1,
    "semantic_xpath_ms": 5713.9,
    "total_ms": 9545.1
  },
  "tree": {
    "before": { ... },
    "after": { ... }
  }
}
```

### Get Tree State

```bash
curl http://localhost:5000/api/tree
```

### Reset Tree

```bash
curl -X POST http://localhost:5000/api/tree/reset
```

### CORS Configuration

The API supports CORS for frontend integration (React, Vue, etc.):

- Allowed origins: `localhost:5173`, `localhost:3000`, `127.0.0.1:5173`
- Allowed methods: `GET`, `POST`, `PUT`, `DELETE`, `OPTIONS`

---

## Mathematical Foundation

### Data Model

Hierarchical data is a rooted tree `T = (V, E, r)`:
- `V`: nodes (e.g., Day, POI, Restaurant)
- `E`: parent-child edges
- `r`: root node

Each node `v` has:
- Type `κ(v)` (e.g., "POI", "Day")
- Content `X_v` (text fields like name, description)
- Children `ch(v)` (structural children defined in schema)

### Semantic Scoring

For a semantic condition `c` (e.g., "museum") and node `v` with content `X_v`:

```
π_v(c) = P(node v satisfies c | X_v)
```

This posterior probability is computed by a scorer (LLM, NLI entailment, or cosine similarity).

### Three Predicate Types

Following the paper formalization, we define recursive predicate scoring `Score(u, ψ)`:

#### 1. atom() - Atomic Predicate (Local Semantic Match)

Scores the node's **own content only**. Does not look at children.

```
POI[atom(content =~ "museum")]

Atom(u, φ) = Scorer(attr(u), φ)  where attr(u) = node's text content
```

#### 2. agg_exists() - Existential Aggregation (Max)

"At least one child matches" - high score if ANY child has high score.

```
Day[agg_exists(POI[atom(content =~ "museum")])]

Score(u, ψ) = Agg∃({Score(c, ψ') | c ∈ children(u)}) = max(...)
```

**Example**: Children with scores [0.95, 0.1, 0.05]
```
agg_exists() = max(0.95, 0.1, 0.05) = 0.95
```

#### 3. agg_prev() - Prevalence Aggregation (Average)

"Children are generally X" - average of child scores.

```
Day[agg_prev(POI[atom(content =~ "artistic")])]

Score(u, ψ) = Aggprev({Score(c, ψ') | c ∈ children(u)}) = mean(...)
```

**Example**: Children with scores [0.8, 0.7, 0.6, 0.3]
```
agg_prev() = (0.8 + 0.7 + 0.6 + 0.3) / 4 = 2.4 / 4 = 0.6
```

### Logical Operators

#### AND (Conjunction - Min)

```
atom(content =~ "outdoor") AND atom(content =~ "historic")

Score(u, ψ₁ ∧ ψ₂) = min(Score(u, ψ₁), Score(u, ψ₂))
```

Both conditions must be satisfied - the score is limited by the weakest match.

#### OR (Disjunction - Max)

```
atom(content =~ "museum") OR atom(content =~ "gallery")

Score(u, ψ₁ ∨ ψ₂) = max(Score(u, ψ₁), Score(u, ψ₂))
```

#### NOT (Negation - Complement)

```
not(atom(content =~ "expensive"))

Score(u, ¬ψ) = 1 - Score(u, ψ)
```

Inverts the score - high match becomes low, low match becomes high. Use for exclusion queries like "not work related" or "not expensive".

### Score Fusion Across Steps

For multi-step queries, scores are multiplied:

```
Query: /Day[agg_prev(...)]/POI[atom(...)]

For each final POI node u:
  Final Score = ∏_{steps with predicates} Score(u, ψ_i)
```

This ensures:
- High parent score + high child score → high final score
- Low parent score penalizes even high-scoring children
- Simple and interpretable score combination

---

## Query Syntax

### Path Navigation

```xpath
/Itinerary/Day/POI                     # All POIs in all Days
/Itinerary/Day[2]/POI                  # POIs in Day 2 (positional index)
/Itinerary/Day/POI[2]                  # 2nd POI in EACH Day (local)
(/Itinerary/Day/POI)[2]                # 2nd POI overall (global)
```

### Positional Indexing

| Syntax | Meaning |
|--------|---------|
| `[2]` | Select node at index 2 |
| `[2]` | 2nd element |
| `[-1]` | Last element |
| `[1:3]` | Elements 1, 2, 3 |
| `[-2:]` | Last 2 elements |

### Semantic Predicates

```xpath
# Local match (node's own content)
/Itinerary/Day/POI[atom(content =~ "museum")]

# Existential (any child matches)
/Itinerary/Day[agg_exists(POI[atom(content =~ "museum")])]

# Prevalence (children generally match)
/Itinerary/Day[agg_prev(POI[atom(content =~ "artistic")])]

# Logical operators
/Itinerary/Day/POI[atom(content =~ "outdoor") AND atom(content =~ "free")]
/Itinerary/Day/POI[atom(content =~ "museum") OR atom(content =~ "gallery")]

# Negation - exclude matches
/Itinerary/Day/POI[not(atom(content =~ "expensive"))]
/Itinerary/Day/POI[not(atom(content =~ "work related"))]

# Combined AND with NOT
/Itinerary/Day/POI[atom(content =~ "museum") AND not(atom(content =~ "expensive"))]

# NOT with aggregation
/Itinerary/Day[not(agg_exists(Restaurant[atom(content =~ "upscale")]))]

# Aggregation-level AND/OR
/Itinerary/Day[agg_exists(POI[atom(content =~ "museum")]) AND agg_exists(Restaurant[atom(content =~ "italian")])]
```

---

## Predicate Types

### Decision Guide

| Query Intent | Predicate | Example |
|--------------|-----------|---------|
| Property of the node itself | `atom()` | "museum POI" → `POI[atom(...)]` |
| Any child has property X | `agg_exists()` | "day with a museum" → `Day[agg_exists(POI[...])]` |
| Children are generally X | `agg_prev()` | "artistic day" → `Day[agg_prev(POI[...])]` |

### Comparison

```
Day with 5 POIs: [Museum: 0.95, Park: 0.1, Mall: 0.05, Theater: 0.1, Cafe: 0.02]

agg_exists(POI[atom(content =~ "museum")]): 0.95 (max - museum exists)
agg_prev(POI[atom(content =~ "museum")]):   0.244 (avg - most aren't museums)
```

---

## Schema System

Schemas define the tree structure and distinguish node fields from children.

### Example Schema (itinerary.yaml)

```yaml
name: "itinerary"

nodes:
  Itinerary:
    type: root
    fields: []
    children: ["Version"]           # Root contains Version nodes
    
  Version:
    type: container
    index_attr: "number"            # <Version number="1">
    fields:
      - patch_info                  # Description of changes
      - conversation_history        # User's original request
    children: ["Day"]
    
  Day:
    type: container
    index_attr: "index"
    fields: []                      # Day's own fields
    children: ["POI", "Restaurant"] # Structural children
    
  POI:
    type: leaf
    fields: [name, time_block, description, travel_method, expected_cost, highlights]
    children: []
    
  Restaurant:
    type: leaf
    fields: [name, time_block, description, travel_method, expected_cost, highlights]
    children: []
```

### Why fields vs children matters

```xml
<Day index="1">
  <POI>...</POI>          <!-- Child: in children list -->
  <Restaurant>...</Restaurant>  <!-- Child: in children list -->
  <theme>relaxing</theme>  <!-- Field: NOT in children list, part of Day's own content -->
</Day>
```

When scoring `Day[agg_prev(POI[...])]`:
- `<POI>` and `<Restaurant>` are structural children → aggregated
- `<theme>` is Day's own field → not treated as child

### Available Schemas

| Schema | Hierarchy | Use Case |
|--------|-----------|----------|
| `itinerary` | Itinerary → Day → POI/Restaurant | Travel planning |
| `todolist` | TodoList → Project → Task → SubTask | Task management |
| `curriculum` | Curriculum → Course → Concept/Exercise | Education |
| `support` | SupportSystem → Customer → Ticket → Symptom/Cause/Resolution | Help desk |
| `session_recommendation` | RecommendationHub → Session → Step → Objective → Item | Shopping/DIY |

---

## Usage

### CLI Interactive Mode

```bash
python -m pipeline.semantic_xpath_pipeline
```

**Example Session:**

```
============================================================
Semantic XPath Pipeline - CRUD Operations
============================================================
In-tree versioning enabled:
  - All modifications create new versions
  - Query specific versions with Version[N] or Version[-1]
  - Search versions: 'what changed about museum?'
------------------------------------------------------------
Commands:
  - Natural language query for CRUD operations
  - 'stats' - Session statistics
  - 'history' - View version history
  - 'reload' - Reload tree from file
  - 'exit' or 'quit' - Exit
============================================================

🔄 Query: delete the museum

📋 Delete(/Itinerary/Version[-1]/Day/POI[atom(content =~ "museum")])
📁 Tree saved: result/demo/travel_memory_3day.xml (2 versions)

⏱️  Step Timing:
---------------------------------------------
  Query Generation             621.9ms  █░░░░░░░░░░░░░░░░░░░   6.5%
  Version Resolution             1.5ms  ░░░░░░░░░░░░░░░░░░░░   0.0%
  Semantic XPath Execution    5713.9ms  ███████████░░░░░░░░░  59.9%
  LLM Node Reasoning          3205.7ms  ██████░░░░░░░░░░░░░░  33.6%
  ...
---------------------------------------------
  TOTAL                       9545.1ms

✅ DELETE Operation Succeeded - Created Version 2
============================================================

🔄 Query: in the version that deleted museums, update first POI to chinese food

📋 Update(/Itinerary/Version[atom(content =~ "delete museum")]/Day/POI[1], type: chinese food)
📁 Tree saved: result/demo/travel_memory_3day.xml (3 versions)

✅ UPDATE Operation Succeeded (Version 2 → Version 3)
```

### CLI Options

```bash
# Scoring methods
python -m pipeline.semantic_xpath_pipeline --scoring llm        # GPT-4 scoring (highest accuracy)
python -m pipeline.semantic_xpath_pipeline --scoring entailment # BART NLI (balanced)
python -m pipeline.semantic_xpath_pipeline --scoring cosine     # Embedding similarity (fastest)

# Threshold and top-k
python -m pipeline.semantic_xpath_pipeline --top-k 10 --threshold 0.5

# Single query mode (non-interactive)
python -m pipeline.semantic_xpath_pipeline -q "find museums"
```

### Programmatic Usage

```python
# Full CRUD Pipeline
from pipeline import SemanticXPathPipeline

pipeline = SemanticXPathPipeline(
    scoring_method="entailment",
    top_k=10,
    score_threshold=0.3
)

# Any CRUD operation via natural language
result = pipeline.process_request("delete all the museums")
print(pipeline.format_result(result))

# Direct XPath execution (read-only)
from pipeline_execution.semantic_xpath_execution import DenseXPathExecutor

executor = DenseXPathExecutor(
    schema_name="itinerary",
    scoring_method="entailment",
    top_k=5,
    score_threshold=0.3
)

# Simple semantic query
result = executor.execute('/Itinerary/Day/POI[atom(content =~ "museum")]')

# Hierarchical query with aggregation
result = executor.execute(
    '/Itinerary/Day[agg_prev(POI[atom(content =~ "artistic")])]/POI[atom(content =~ "museum")]'
)

for node in result.matched_nodes:
    print(f"- {node.tree_path}: {node.score:.3f}")
```

---

## Configuration

Edit `config.yaml`:

```yaml
# Schema selection
active_schema: "itinerary"
active_data: "travel_memory_3day"

# Executor settings
xpath_executor:
  top_k: 5
  score_threshold: 0.01
  scoring_method: "entailment"  # "llm", "entailment", or "cosine"

# OpenAI (for LLM scoring and query generation)
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o"
```

### Scoring Methods

| Method | Speed | Accuracy | Cost |
|--------|-------|----------|------|
| `llm` | Slow | Highest | API costs |
| `entailment` | Medium | High | Free (local) |
| `cosine` | Fast | Good | Free (local) |

---

## Project Structure

```
LLM-VM/
├── api/                             # REST API (Flask)
│   ├── app.py                       # Flask app factory
│   ├── run.py                       # Server entry point
│   └── routes/
│       ├── query.py                 # POST /api/query
│       ├── tree.py                  # GET/POST /api/tree/*
│       └── config.py                # GET/PUT /api/config
├── pipeline/
│   ├── semantic_xpath_pipeline.py   # Main entry point (CRUD pipeline)
│   └── serializers.py               # JSON serialization for API
├── pipeline_execution/
│   ├── semantic_xpath_orchestrator.py  # CRUD operation orchestrator
│   ├── base.py                      # Base handler classes and result types
│   ├── prompt_loader.py             # Dynamic prompt composition
│   └── crud/                        # CRUD handlers
│       ├── read_handler.py          # READ operation handler
│       ├── delete_handler.py        # DELETE operation handler
│       ├── update_handler.py        # UPDATE operation handler
│       └── create_handler.py        # CREATE operation handler
├── reasoner/
│   ├── base.py                      # ReasonerDecision, InsertionPoint
│   ├── node_reasoner.py             # Batched LLM node selection
│   └── insertion_reasoner.py        # Find insertion points for Create
├── tree_modification/
│   ├── base.py                      # OperationResult, path utilities
│   ├── node_deleter.py              # Delete nodes by path
│   ├── node_inserter.py             # Insert/replace nodes
│   └── version_manager.py           # In-tree versioning (create/resolve versions)
├── content_creator/
│   ├── base.py                      # ContentGenerationResult
│   ├── node_creator.py              # LLM content generation
│   └── node_updater.py              # LLM content updates
├── xpath_query_generation/
│   └── xpath_query_generator.py     # Unified: NL → CRUD + XPath (single LLM call)
├── dense_xpath/
│   ├── dense_xpath_executor.py      # Main executor with score fusion
│   ├── models.py                    # AtomicPredicate, CompoundPredicate, etc.
│   ├── parser.py                    # Query parser (atom/agg_exists/agg_prev/AND/OR/NOT)
│   ├── predicate_handler.py         # Scoring + aggregation logic
│   ├── node_utils.py                # XML node utilities
│   ├── schema_loader.py             # Schema loading
│   └── trace_writer.py              # Execution + CRUD traces
├── predicate_classifier/
│   ├── llm_scorer.py                # GPT-4 scoring
│   ├── entailment_scorer.py         # BART NLI scoring (used for version resolution)
│   └── cosine_scorer.py             # Embedding similarity
├── client/                          # Model clients
│   ├── openai_client.py             # OpenAI API wrapper
│   ├── bart_client.py               # BART NLI client
│   └── tas_b_client.py              # TAS-B embedding client
├── storage/
│   ├── schemas/                     # Schema definitions (itinerary, todolist, etc.)
│   ├── memory/                      # XML data files (versioned structure)
│   └── prompts/                     # All LLM prompts
├── result/
│   └── demo/                        # Modified trees (all versions in one file)
├── traces/                          # Execution logs and reasoning traces
├── tests/                           # Test suite
├── config.yaml                      # Main configuration
├── requirements.txt                 # Python dependencies
├── framework.md                     # Mathematical specification
└── README.md
```

---

## Quick Reference

### Version Selectors

| Selector | Meaning | Example |
|----------|---------|---------|
| `Version[-1]` | Latest version (default) | `/Itinerary/Version[-1]/Day/POI` |
| `Version[N]` | Specific version | `/Itinerary/Version[2]/Day` |
| `Version[atom(...)]` | Semantic search | `/Itinerary/Version[atom(content =~ "delete museum")]/Day` |

### CRUD Operations

| Operation | Natural Language Example | Full Query |
|-----------|-------------------------|------------|
| **Read** | "find museums" | `Read(/Itinerary/Version[-1]/Day/POI[atom(...)])` |
| **Create** | "add a cafe on day 1" | `Create(/Itinerary/Version[-1]/Day[1], Restaurant, ...)` |
| **Update** | "change CN Tower to 2pm" | `Update(/Itinerary/Version[-1]/Day/POI[atom(...)], time_block: 2:00 PM)` |
| **Compound** | "in version that deleted museum, update first POI" | `Update(/Itinerary/Version[atom(content =~ "delete museum")]/Day/POI[1], ...)` |
| **Delete** | "remove all museums" | `Delete(/Itinerary/Version[-1]/Day/POI[atom(...)])` |

### Predicate Syntax

| Predicate | Syntax | Use When |
|-----------|--------|----------|
| `atom()` | `atom(content =~ "X")` | Matching node's own content |
| `agg_exists()` | `agg_exists(Child[atom(...)])` | Any child has property |
| `agg_prev()` | `agg_prev(Child[atom(...)])` | Children generally have property |
| `not()` | `not(atom(...))` | Excluding matches (negation) |

### Aggregation Formulas

| Operator | Formula | Interpretation |
|----------|---------|----------------|
| `atom()` | `Atom(u, φ) = Scorer(attr(u), φ)` | Local score |
| `agg_exists()` | `Agg∃(A) = max(A)` | Max over children |
| `agg_prev()` | `Aggprev(A) = mean(A)` | Average over children |
| `AND` | `Score(u, ψ₁ ∧ ψ₂) = min(...)` | Min of scores |
| `OR` | `Score(u, ψ₁ ∨ ψ₂) = max(...)` | Max of scores |
| `NOT` | `Score(u, ¬ψ) = 1 - Score(u, ψ)` | Complement (inversion) |

### Example Queries

```xpath
# Find museums (latest version)
/Itinerary/Version[-1]/Day/POI[atom(content =~ "museum")]

# Find days with museums
/Itinerary/Version[-1]/Day[agg_exists(POI[atom(content =~ "museum")])]

# Find artistic days
/Itinerary/Version[-1]/Day[agg_prev(POI[atom(content =~ "artistic")])]

# Find museums in artistic days
/Itinerary/Version[-1]/Day[agg_prev(POI[atom(content =~ "artistic")])]/POI[atom(content =~ "museum")]

# Days with both museum AND Italian restaurant
/Itinerary/Version[-1]/Day[agg_exists(POI[atom(content =~ "museum")]) AND agg_exists(Restaurant[atom(content =~ "italian")])]

# Find POIs that are NOT expensive
/Itinerary/Version[-1]/Day/POI[not(atom(content =~ "expensive"))]

# Find museums that are NOT expensive (AND + NOT)
/Itinerary/Version[-1]/Day/POI[atom(content =~ "museum") AND not(atom(content =~ "expensive"))]

# Days without upscale restaurants (NOT with aggregation)
/Itinerary/Version[-1]/Day[not(agg_exists(Restaurant[atom(content =~ "upscale")]))]

# Query specific version by semantic search
/Itinerary/Version[atom(content =~ "delete museum")]/Day/POI
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `SemanticXPathPipeline` | Main entry point for CRUD operations |
| `SemanticXPathOrchestrator` | Orchestrates CRUD operations with versioning |
| `XPathQueryGenerator` | Unified LLM call: NL → CRUD + XPath |
| `VersionManager` | In-tree versioning: resolve/create versions |
| `DenseXPathExecutor` | Core XPath execution with semantic scoring |
| `NodeReasoner` | LLM selects relevant nodes from candidates |
| `NodeCreator` / `NodeUpdater` | LLM generates/modifies node content |
| `EntailmentScorer` | BART NLI scoring for version resolution |

---

## License

MIT License

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

```bash
# Run tests
pytest tests/

# Run a single test
pytest tests/test_query_generation.py -v

# Run NOT and AND operator tests
python tests/test_not_and_operators.py
```
