"""
Analyze semantic_xpath pipeline stages for timing and token usage.
"""
import json
import glob
from pathlib import Path

def analyze_stages(experiment_name: str):
    base_dir = Path(__file__).parent / "experiment_results" / experiment_name / "semantic_xpath"
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Collect all stage data
    all_stages = {}
    query_count = 0
    
    result_files = sorted(glob.glob(str(base_dir / "query_*" / "result.json")))
    
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_result = data.get("raw_result", {})
        timing = raw_result.get("timing", {})
        stages = timing.get("stages", [])
        
        if not stages:
            continue
            
        query_count += 1
        
        for stage in stages:
            name = stage.get("name", "unknown")
            time_ms = stage.get("time_ms", 0)
            token_usage = stage.get("token_usage", {})
            
            if name not in all_stages:
                all_stages[name] = {
                    "total_time_ms": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "count": 0
                }
            
            all_stages[name]["total_time_ms"] += time_ms
            all_stages[name]["count"] += 1
            if token_usage:
                all_stages[name]["total_prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                all_stages[name]["total_completion_tokens"] += token_usage.get("completion_tokens", 0)
                all_stages[name]["total_tokens"] += token_usage.get("total_tokens", 0)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Semantic XPath Pipeline Stage Analysis: {experiment_name}")
    print(f"Queries analyzed: {query_count}")
    print(f"{'='*80}\n")
    
    # Order stages
    stage_order = ["version_resolution", "version_lookup", "query_generation", "xpath_execution", "downstream_task"]
    
    # Table header
    print(f"{'Stage':<25} {'Time (s)':<12} {'Time %':<10} {'Prompt':<10} {'Completion':<12} {'Total Tok':<10}")
    print("-" * 80)
    
    total_time = sum(s["total_time_ms"] for s in all_stages.values())
    total_prompt = sum(s["total_prompt_tokens"] for s in all_stages.values())
    total_completion = sum(s["total_completion_tokens"] for s in all_stages.values())
    total_tokens = sum(s["total_tokens"] for s in all_stages.values())
    
    for stage_name in stage_order:
        if stage_name not in all_stages:
            continue
        stage = all_stages[stage_name]
        time_s = stage["total_time_ms"] / 1000
        time_pct = (stage["total_time_ms"] / total_time * 100) if total_time > 0 else 0
        prompt = stage["total_prompt_tokens"]
        completion = stage["total_completion_tokens"]
        tokens = stage["total_tokens"]
        
        print(f"{stage_name:<25} {time_s:>10.2f}s {time_pct:>8.1f}% {prompt:>10,} {completion:>12,} {tokens:>10,}")
    
    print("-" * 80)
    print(f"{'TOTAL':<25} {total_time/1000:>10.2f}s {'100.0%':>9} {total_prompt:>10,} {total_completion:>12,} {total_tokens:>10,}")
    
    # Average per query
    print(f"\n{'='*80}")
    print("AVERAGES PER QUERY")
    print(f"{'='*80}\n")
    
    print(f"{'Stage':<25} {'Avg Time':<12} {'Avg Prompt':<12} {'Avg Completion':<15} {'Avg Total':<10}")
    print("-" * 80)
    
    for stage_name in stage_order:
        if stage_name not in all_stages:
            continue
        stage = all_stages[stage_name]
        count = stage["count"]
        if count == 0:
            continue
        avg_time = stage["total_time_ms"] / count / 1000
        avg_prompt = stage["total_prompt_tokens"] / count
        avg_completion = stage["total_completion_tokens"] / count
        avg_tokens = stage["total_tokens"] / count
        
        print(f"{stage_name:<25} {avg_time:>10.2f}s {avg_prompt:>12,.0f} {avg_completion:>15,.0f} {avg_tokens:>10,.0f}")
    
    print("-" * 80)
    print(f"{'TOTAL PER QUERY':<25} {total_time/query_count/1000:>10.2f}s {total_prompt/query_count:>12,.0f} {total_completion/query_count:>15,.0f} {total_tokens/query_count:>10,.0f}")

if __name__ == "__main__":
    import sys
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "gpt5mini_experiment"
    analyze_stages(experiment_name)
