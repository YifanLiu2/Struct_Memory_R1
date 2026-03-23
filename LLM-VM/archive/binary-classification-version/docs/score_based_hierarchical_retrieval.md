# Score-Based Hierarchical Retrieval System

## Overview

This document describes the architecture for a score-based hierarchical semantic retrieval system. Instead of binary classification (match/no match), the system propagates continuous scores through the tree hierarchy using MIN/MAX aggregation rules, then applies ranking and optional ordinal post-processing.

---

## 1. Core Concepts

### 1.1 Current System (Binary)
```
User Query → XPath Query → Binary Classification → Filtered Results
```

### 1.2 New System (Score-Based)
```
User Query → XPath Query → Score Collection → Aggregation → Ranking → Results
```

### 1.3 Key Differences

| Aspect | Binary System | Score-Based System |
|--------|--------------|-------------------|
| Classification | `is_match: true/false` | `score: 0.0 - 1.0` |
| Filtering | Hard cutoff at predicate | Soft ranking at end |
| Parent nodes | Binary filter by children | Aggregate children scores |
| Output | Filtered list | Ranked list with scores |

---

## 2. Tree Structure

```
Itinerary (root)
├── Day[1]
│   ├── POI[1]      (leaf)
│   ├── POI[2]      (leaf)
│   ├── Restaurant[1] (leaf)
│   └── Restaurant[2] (leaf)
├── Day[2]
│   ├── POI[1]
│   └── Restaurant[1]
└── Day[3]
    ├── POI[1]
    └── Restaurant[1]
```

**Key Insight**: All semantic information (name, description) lives at leaf nodes. Parent nodes (Day) derive their semantic meaning from their children.

---

## 3. Query Types

### 3.1 Type Matching (Structural)
```xpath
/Itinerary/Day/POI
```
No semantic filtering, just structural traversal.

### 3.2 Index Matching (Structural)
```xpath
/Itinerary/Day[1]/POI
```
Positional selection within a level.

### 3.3 Predicate Matching (Semantic)
```xpath
/Itinerary/Day/POI[description =~ "museum"]
```
Semantic scoring against a predicate.

### 3.4 Ordinal Selection (Post-Processing) - NEW
```
"find me the second artistic day"
```
Applied after ranking, not in XPath syntax.

---

## 4. Score Aggregation Rules

### 4.1 Rule Determination

The aggregation method is **inferred by the executor** based on predicate position:

| Predicate On | Return Level | Aggregation | Rationale |
|--------------|--------------|-------------|-----------|
| Leaf | Leaf | None | Direct score |
| Parent | Leaf | MAX | "Day with museum" → any museum suffices |
| Parent | Parent | MIN/MEAN | "Artistic day" → overall day character |

### 4.2 Examples

#### Example A: "Day with museum"
```xpath
/Itinerary/Day/POI[description =~ "museum"]
```
- Score each POI for "museum"
- Return POIs sorted by museum score
- **Aggregation**: None (returning leaves)

#### Example B: "Artistic day with museum"  
```xpath
/Itinerary/Day[description =~ "artistic"]/POI[description =~ "museum"]
```
- Score each POI for "museum" AND "artistic"
- For "museum" at Day level: **MAX** (just need one museum)
- For "artistic" at Day level: **MIN/MEAN** (overall day vibe)
- Return POIs, but filter by Day-level aggregated scores

#### Example C: "Find artistic days"
```xpath
/Itinerary/Day[description =~ "artistic"]
```
- Score each POI for "artistic"
- Aggregate to Day using **MIN/MEAN**
- Return Days ranked by aggregated score

### 4.3 Aggregation Methods

```python
def aggregate_max(scores: list[float]) -> float:
    """At least one child matches well."""
    return max(scores) if scores else 0.0

def aggregate_min(scores: list[float]) -> float:
    """All children should match."""
    return min(scores) if scores else 0.0

def aggregate_mean(scores: list[float]) -> float:
    """Average child matching (recommended for most cases)."""
    return sum(scores) / len(scores) if scores else 0.0
```

---

## 5. Score Fusion

When a node has multiple predicates, scores must be fused:

### 5.1 Fusion Methods

```python
def fuse_product(scores: dict[str, float]) -> float:
    """Multiplicative fusion - all predicates must score well."""
    result = 1.0
    for score in scores.values():
        result *= score
    return result

def fuse_min(scores: dict[str, float]) -> float:
    """Bottleneck fusion - weakest predicate dominates."""
    return min(scores.values()) if scores else 0.0

def fuse_weighted_sum(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted combination."""
    total = sum(scores[k] * weights.get(k, 1.0) for k in scores)
    return total / sum(weights.get(k, 1.0) for k in scores)
```

### 5.2 Default Strategy

Use **product fusion** as default:
- "Artistic day with museum" → `artistic_score * museum_score`
- Penalizes nodes that fail any predicate

---

## 6. Data Structures

### 6.1 ScoredNode

```python
@dataclass
class ScoredNode:
    """A node with associated scores."""
    node_info: NodeInfo
    scores: dict[str, float]      # predicate -> score
    aggregated_score: float       # final fused score
    layer_index: int              # position in layer (1-based)
    children_scores: Optional[dict[str, list[float]]] = None  # for tracing
```

### 6.2 RankingResult

```python
@dataclass
class RankingResult:
    """Result of ranking operation."""
    ranked_nodes: list[ScoredNode]
    total_count: int
    applied_ordinal: Optional[str] = None  # "first", "second", etc.
    applied_top_k: Optional[int] = None
```

---

## 7. Execution Flow

### 7.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PARSE QUERY                                                  │
│    Input: "/Itinerary/Day[desc=~'artistic']/POI[desc=~'museum']"│
│    Output: [                                                    │
│      Segment(type=Day, predicate='artistic'),                   │
│      Segment(type=POI, predicate='museum')                      │
│    ]                                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. TYPE/INDEX PRUNING                                           │
│    Traverse tree, collect candidate nodes at each level         │
│    Apply index filters (Day[1], Day[2], etc.)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. LEAF NODE SCORING                                            │
│    For each leaf node, score against ALL predicates in query    │
│    POI1: {museum: 0.9, artistic: 0.7}                           │
│    POI2: {museum: 0.2, artistic: 0.9}                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. SCORE AGGREGATION                                            │
│    Propagate leaf scores to parent nodes:                       │
│    Day1.museum = MAX(POI1.museum, POI2.museum) = 0.9            │
│    Day1.artistic = MEAN(POI1.artistic, POI2.artistic) = 0.8     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. SCORE FUSION                                                 │
│    Combine predicate scores into single score:                  │
│    Day1.final = Day1.museum * Day1.artistic = 0.72              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. RANKING                                                      │
│    Sort nodes by aggregated_score descending                    │
│    Apply top-k if specified                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. ORDINAL POST-PROCESSING (Optional)                           │
│    "second" → return node at index 1                            │
│    "last" → return node at index -1                             │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Detailed Algorithm

```python
def execute_score_based(query: str) -> RankingResult:
    # 1. Parse
    segments = parse_query(query)
    
    # 2. Collect nodes by level with pruning
    nodes_by_level = collect_nodes_with_index(segments)
    
    # 3. Identify all predicates
    predicates = [(seg.predicate, seg_idx) for seg_idx, seg in enumerate(segments) if seg.predicate]
    
    # 4. Score leaf nodes for ALL predicates
    leaf_level = max(nodes_by_level.keys())
    leaf_nodes = nodes_by_level[leaf_level]
    
    scored_leaves = []
    for node, layer_idx in leaf_nodes:
        scores = {}
        for predicate, _ in predicates:
            scores[predicate] = classifier.score(node, predicate)
        scored_leaves.append(ScoredNode(node, scores, layer_index=layer_idx))
    
    # 5. Determine return level (deepest segment)
    return_level = len(segments) - 1
    
    # 6. Aggregate scores to return level
    if return_level < leaf_level:
        # Need to aggregate from leaves to parent
        scored_parents = aggregate_to_level(scored_leaves, return_level, segments)
    else:
        scored_parents = scored_leaves
    
    # 7. Fuse scores
    for node in scored_parents:
        node.aggregated_score = fuse_product(node.scores)
    
    # 8. Rank
    ranked = sorted(scored_parents, key=lambda n: n.aggregated_score, reverse=True)
    
    return RankingResult(ranked_nodes=ranked, total_count=len(ranked))
```

---

## 8. Implementation Phases

### Phase 1: Data Structures
- [ ] Add `ScoredNode` to `predicate_classifiers/base.py`
- [ ] Add `RankingResult` dataclass
- [ ] Modify `ClassificationResult` to emphasize score over boolean

### Phase 2: Score Aggregation Module
- [ ] Create `dense_xpath/score_aggregation.py`
- [ ] Implement `aggregate_max`, `aggregate_min`, `aggregate_mean`
- [ ] Implement `fuse_product`, `fuse_min`, `fuse_weighted_sum`

### Phase 3: Ranking Module  
- [ ] Create `dense_xpath/ranking.py`
- [ ] Implement `RankingLayer.rank()`
- [ ] Implement `RankingLayer.top_k()`
- [ ] Implement `RankingLayer.apply_ordinal()`

### Phase 4: Refactor Executor
- [ ] Update `DenseXPathExecutor` for score-based flow
- [ ] Implement aggregation inference logic
- [ ] Track layer indices
- [ ] Return `RankingResult` instead of filtered list

### Phase 5: Unify Classifiers
- [ ] Ensure all classifiers return consistent score format
- [ ] Remove threshold-based `is_match` from classification step
- [ ] Move thresholding to ranking layer

### Phase 6: Update Pipeline
- [ ] Update `xpath_pipeline.py` for new result format
- [ ] Add ordinal extraction from user query
- [ ] Update CLI output for ranked results

---

## 9. Configuration

### 9.1 config.yaml additions

```yaml
score_aggregation:
  # Aggregation method for parent predicates
  parent_predicate_method: "mean"  # min, max, mean
  
  # Aggregation method for child predicates propagated to parent
  child_to_parent_method: "max"    # min, max, mean
  
  # Score fusion method
  fusion_method: "product"         # product, min, weighted_sum

ranking:
  # Default top-k if not specified
  default_top_k: 10
  
  # Minimum score threshold (optional hard cutoff)
  min_score_threshold: 0.0
```

---

## 10. Example Traces

### Example: "Find artistic day with museums"

**Input Query**: `/Itinerary/Day[description =~ "artistic"]/POI[description =~ "museum"]`

**Step 1: Parse**
```json
{
  "segments": [
    {"type": "Day", "predicate": "artistic"},
    {"type": "POI", "predicate": "museum"}
  ]
}
```

**Step 2: Collect Nodes**
```json
{
  "Day": [
    {"path": "/Itinerary/Day[1]", "layer_index": 1},
    {"path": "/Itinerary/Day[2]", "layer_index": 2},
    {"path": "/Itinerary/Day[3]", "layer_index": 3}
  ],
  "POI": [
    {"path": "/Itinerary/Day[1]/POI[1]", "layer_index": 1, "name": "St. Lawrence Market"},
    {"path": "/Itinerary/Day[1]/POI[2]", "layer_index": 2, "name": "CN Tower"},
    {"path": "/Itinerary/Day[1]/POI[3]", "layer_index": 3, "name": "Art Gallery of Ontario"},
    ...
  ]
}
```

**Step 3: Score Leaves**
```json
{
  "/Itinerary/Day[1]/POI[1]": {"museum": 0.3, "artistic": 0.2},
  "/Itinerary/Day[1]/POI[2]": {"museum": 0.1, "artistic": 0.3},
  "/Itinerary/Day[1]/POI[3]": {"museum": 0.9, "artistic": 0.95},
  "/Itinerary/Day[2]/POI[1]": {"museum": 0.95, "artistic": 0.8},
  ...
}
```

**Step 4: Aggregate to Day**
```json
{
  "/Itinerary/Day[1]": {
    "museum": 0.9,      // MAX of children
    "artistic": 0.48    // MEAN of children
  },
  "/Itinerary/Day[2]": {
    "museum": 0.95,
    "artistic": 0.75
  }
}
```

**Step 5: Fuse & Rank**
```json
{
  "ranked": [
    {"path": "/Itinerary/Day[2]", "score": 0.7125},  // 0.95 * 0.75
    {"path": "/Itinerary/Day[1]", "score": 0.432}    // 0.9 * 0.48
  ]
}
```

---

## 11. Open Questions

1. **Top-K determination**: How to automatically determine K when user doesn't specify?
   - Option A: Use confidence gap (big drop in scores)
   - Option B: Fixed default (e.g., top 3)
   - Option C: Return all with scores, let consumer decide

2. **Ordinal parsing**: Should "second artistic day" be parsed in query generator or separately?
   - Decision: Post-processing (already decided)

3. **Mixed return levels**: What if query has predicates at multiple levels and we want to return intermediate level?
   - Current: Return deepest level by default
   - Future: Allow explicit return level specification

---

## 12. Testing Scenarios

| Query | Expected Behavior |
|-------|-------------------|
| "Find museums" | Score POIs for "museum", rank by score |
| "Find artistic day" | Score POIs for "artistic", aggregate to Day with MEAN, rank Days |
| "Find museum in artistic day" | Score POIs for both, aggregate "artistic" to Day, return POIs ranked |
| "Find the second artistic day" | Rank Days by "artistic", return 2nd |
| "Find Italian restaurant in relaxing day" | Score Restaurants for "Italian", aggregate "relaxing" to Day, rank |

