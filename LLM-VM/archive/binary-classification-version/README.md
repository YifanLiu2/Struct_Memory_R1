# LLM-VM: Semantic XPath Query Engine

A semantic query engine that converts natural language queries into XPath-like expressions and executes them against structured tree data using LLM-based predicate classification.

## Overview

This project enables semantic querying of hierarchical data structures (like travel itineraries) using natural language. It combines:

1. **XPath-like Query Generation**: Converts natural language to structured queries
2. **Dense XPath Execution**: Traverses XML trees with semantic predicate matching
3. **Multi-Strategy Classification**: Choose from LLM, entailment, or cosine similarity

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     XPathPipeline                           │
│  ┌──────────────────────┐    ┌──────────────────────────┐   │
│  │  XPathQueryGenerator │    │   DenseXPathExecutor     │   │
│  │  (NL → XPath)        │───▶│   (XPath → Results)      │   │
│  └──────────────────────┘    └───────────┬──────────────┘   │
└──────────────────────────────────────────┼──────────────────┘
                                           │
                      ┌────────────────────┼────────────────────┐
                      │                    │                    │
                      ▼                    ▼                    ▼
              ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
              │     LLM      │   │  Entailment  │   │    Cosine    │
              │  Classifier  │   │  Classifier  │   │  Classifier  │
              │   (GPT-4o)   │   │  (BART-NLI)  │   │   (TAS-B)    │
              └──────────────┘   └──────────────┘   └──────────────┘
```

## Project Structure

```
LLM-VM/
├── client/                      # API clients
│   ├── openai_client.py         # OpenAI API wrapper
│   ├── bart_client.py           # BART NLI client (entailment)
│   └── tas_b_client.py          # TAS-B embedding client (similarity)
│
├── xpath_query_generation/      # Natural language to XPath
│   └── xpath_query_generator.py
│
├── dense_xpath/                 # XPath execution engine
│   └── dense_xpath_executor.py
│
├── predicate_classifiers/       # Semantic classification
│   ├── base.py                  # Abstract base class
│   ├── llm_classifier.py        # LLM-based classifier (GPT-4o)
│   ├── entailment_classifier.py # Entailment-based (BART-NLI)
│   └── cosine_classifier.py     # Cosine similarity (TAS-B)
│
├── pipeline/                    # Main orchestration
│   └── xpath_pipeline.py
│
├── storage/
│   ├── memory/
│   │   └── tree_memory.xml      # Sample itinerary data
│   └── prompts/
│       ├── xpath_query_generator.txt
│       └── predicate_classifier.txt
│
├── reasoning_traces/            # Execution logs and traces
│   ├── logs/                    # Pipeline session logs
│   └── traces/                  # Detailed execution traces (JSON)
│
└── config.yaml                  # Configuration (API keys, etc.)
```

## Tree Hierarchy

The system operates on a tree with the following structure:

```
Itinerary (root)
├── Day
│   ├── POI (Point of Interest)
│   └── Restaurant
```

**Important**: POI and Restaurant are siblings (same level under Day).

## Query Syntax

XPath-like queries with semantic predicates:

```
/Itinerary/Day[1]/Restaurant[description =~ "italian"]
/Itinerary/Day/POI[description =~ "jazz"]
/Itinerary/Day[description =~ "relaxing"]/Restaurant[description =~ "seafood"]
```

### Query Components

| Component | Example | Description |
|-----------|---------|-------------|
| Type match | `/Day` | Match nodes by type |
| Index match | `/Day[1]` | Match by position (1-indexed) |
| Predicate | `[description =~ "italian"]` | Semantic matching |

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LLM-VM
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install openai pyyaml torch transformers
```

4. Configure in `config.yaml`:
```yaml
openai:
  api_key: "your-api-key-here"
  model: "gpt-4o"
  temperature: 0.7
  max_tokens: 4096

entailment:
  threshold: 0.5
  hypothesis_template: "This is related to {predicate}."

cosine_similarity:
  threshold: 0.6
```

## Usage

### Interactive CLI

```bash
# Default (LLM classifier)
python -m pipeline.xpath_pipeline

# With entailment classifier (BART-NLI)
python -m pipeline.xpath_pipeline -c entailment

# With cosine similarity classifier (TAS-B)
python -m pipeline.xpath_pipeline -c cosine
```

```
============================================================
  XPath Pipeline - Interactive Mode
============================================================
  Session ID: 20251222_120000
  Classifier: llm
  Log file: reasoning_traces/logs/pipeline_20251222_120000.log
  Commands:
    quit/exit - Exit and save session
    history   - Show query history
    save      - Save session
    query     - Generate query only (no execution)
============================================================

You: italian restaurants in day 1
----------------------------------------
XPath: /Itinerary/Day[1]/Restaurant[description =~ "italian"]

✓ Found 1 matching node(s):
----------------------------------------
  [1] /Itinerary/Day[1]/Restaurant[2]
      Restaurant: Buca Yorkville
      └─ Upscale Italian dining featuring traditional dishes with a modern twist.
----------------------------------------
```

### Programmatic Usage

```python
from pipeline import XPathPipeline

# Create pipeline with LLM (default)
pipeline = XPathPipeline()

# Or with specific classifier
pipeline = XPathPipeline(classifier_type="entailment")  # BART-NLI
pipeline = XPathPipeline(classifier_type="cosine")      # TAS-B

# Run query
results, xpath_query, trace = pipeline.run("jazz venues in all days")

# Access results
for r in results:
    print(f"{r.tree_path}: {r.name}")
    print(f"  {r.description}")
```

### Query Generation Only

```python
from xpath_query_generation import XPathQueryGenerator

generator = XPathQueryGenerator()
query = generator.generate("cheap restaurants in day 2")
# Output: /Itinerary/Day[2]/Restaurant[description =~ "cheap"]
```

## Execution Algorithm

### Bottom-Up Predicate Processing

For queries with multiple predicates like:
```
/Itinerary/Day[description =~ "relaxing"]/Restaurant[description =~ "jazz"]
```

1. **Top-down traversal**: Type/index matching to prune irrelevant nodes
2. **Bottom-up predicate evaluation**:
   - First process deepest predicate ("jazz" on Restaurants)
   - Find parent nodes (Days) with matching children
   - Evaluate parent predicate ("relaxing") with subtree context
3. **Batch classification**: All nodes at each level are classified together for efficiency

### Subtree Context

When classifying parent nodes (e.g., Day), the classifier receives the full subtree:

```
[0] /Itinerary/Day[3]
Type: Day
Children:
  [1] POI: Toronto Islands - Ferry ride to scenic islands with beaches
  [2] POI: Distillery District - Historic pedestrian village
  [3] Restaurant: Jazz Bistro - Renowned jazz club
```

This allows the LLM to determine if a Day is "relaxing" based on its activities.

## Reasoning Traces

Each execution produces detailed traces in `reasoning_traces/traces/`:

```json
{
  "query": "/Itinerary/Day[description =~ \"relaxing\"]/Restaurant[description =~ \"jazz\"]",
  "timestamp": "2025-12-22T00:10:37",
  "steps": [
    {"step": 1, "action": "type_match", "segment": "/Day", "nodes_before": 1, "nodes_after": 3},
    {"step": 2, "action": "type_match", "segment": "/Restaurant", "nodes_before": 3, "nodes_after": 10},
    {"step": 3, "action": "predicate_batch", "segment": "[description =~ \"jazz\"]", "nodes_before": 10, "nodes_after": 2},
    {"step": 4, "action": "predicate_batch_with_subtree", "segment": "[description =~ \"relaxing\"]", "nodes_before": 2, "nodes_after": 1}
  ],
  "batch_classifications": [
    {
      "predicate": "jazz",
      "input_nodes": [...],
      "results": [
        {"tree_path": "/Itinerary/Day[1]/Restaurant[3]", "is_match": true, "reasoning": "Name contains 'Jazz' and description mentions live jazz performances."},
        {"tree_path": "/Itinerary/Day[3]/Restaurant[4]", "is_match": true, "reasoning": "Named 'Jazz Bistro' - a renowned jazz club."}
      ]
    }
  ],
  "results": [...]
}
```

## Classifier Comparison

| Classifier | Model | Pros | Cons |
|------------|-------|------|------|
| **LLM** | GPT-4o | Most accurate, provides reasoning | Slow, API cost |
| **Entailment** | BART-NLI | Good for logical inference, local | Less flexible |
| **Cosine** | TAS-B | Fast, good for similarity, local | No reasoning |

### LLM Classifier
Uses GPT-4o for semantic classification with detailed reasoning:
```json
{"index": 0, "match": true, "reasoning": "Name contains 'Jazz' and description mentions live performances."}
```

### Entailment Classifier
Uses BART-NLI to score entailment between node text and predicate:
- For leaf nodes: scores node description directly
- For parent nodes: averages scores of children in subtree
- Configurable threshold (default: 0.5)

### Cosine Classifier
Uses TAS-B embeddings for semantic similarity:
- Fast batch processing with normalized embeddings
- For parent nodes: averages similarity scores of children
- Configurable threshold (default: 0.6)

## Extending the System

### Adding New Classifiers

Implement the `PredicateClassifier` base class:

```python
from predicate_classifiers.base import PredicateClassifier, BatchClassificationResult

class MyClassifier(PredicateClassifier):
    def classify_batch(self, nodes, predicate) -> BatchClassificationResult:
        # Your classification logic
        pass
```

Then register in `predicate_classifiers/__init__.py` and `pipeline/xpath_pipeline.py`.

## License

MIT License

