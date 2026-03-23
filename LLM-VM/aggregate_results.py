import json
from pathlib import Path

def main():
    locomo_dir = Path(r"c:\Users\Liam\Desktop\Struct_Memory_R1\LLM-VM\experiment\locomo")
    results_by_conv = {}

    for res_file in sorted(locomo_dir.glob("*/eval_results.json")):
        dir_name = res_file.parent.name
        conv_id = dir_name.split("_")[0]
        
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if conv_id not in results_by_conv:
                results_by_conv[conv_id] = {}
                
            for r in data.get("results", []):
                # Using qa_index to deduplicate/overwrite so resumed runs update old runs safely
                results_by_conv[conv_id][r["qa_index"]] = r

    total_qs = 0
    total_correct = 0
    total_wrong = 0
    total_error = 0
    by_category = {}

    for conv_id, q_dict in results_by_conv.items():
        for qa_idx, r in q_dict.items():
            total_qs += 1
            cat = r.get("category", "unknown")
            verdict = r.get("verdict", "ERROR")
            
            if verdict == "CORRECT": total_correct += 1
            elif verdict == "WRONG": total_wrong += 1
            else: total_error += 1
            
            if cat not in by_category:
                by_category[cat] = {"total": 0, "correct": 0, "wrong": 0, "error": 0}
            
            by_category[cat]["total"] += 1
            if verdict == "CORRECT": by_category[cat]["correct"] += 1
            elif verdict == "WRONG": by_category[cat]["wrong"] += 1
            else: by_category[cat]["error"] += 1

    print("### Aggregated LoCoMo Results\n")
    print(f"**Total Questions Evaluated:** {total_qs}\n")
    if total_qs > 0:
        print(f"- **Correct:** {total_correct} ({total_correct/total_qs*100:.1f}%)")
        print(f"- **Wrong:** {total_wrong} ({total_wrong/total_qs*100:.1f}%)")
        print(f"- **Errors:** {total_error} ({total_error/total_qs*100:.1f}%)\n")

    print("#### Performance by Category")
    print("| Category | Correct | Total | Accuracy |")
    print("|---|---|---|---|")
    for cat, stats in sorted(by_category.items()):
        c = stats["correct"]
        t = stats["total"]
        acc = c / t * 100 if t > 0 else 0
        print(f"| {cat} | {c} | {t} | {acc:.1f}% |")

    print("\n#### Performance by Conversation")
    print("| Conversation | Correct | Total | Accuracy |")
    print("|---|---|---|---|")
    for conv_id, q_dict in sorted(results_by_conv.items()):
        t = len(q_dict)
        c = sum(1 for r in q_dict.values() if r.get("verdict") == "CORRECT")
        acc = c / t * 100 if t > 0 else 0
        print(f"| {conv_id} | {c} | {t} | {acc:.1f}% |")

if __name__ == "__main__":
    main()
