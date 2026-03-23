import json
from pathlib import Path

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
            results_by_conv[conv_id][r["qa_index"]] = r

total_ms = 0
count = 0
for q_dict in results_by_conv.values():
    for r in q_dict.values():
        if "execution_time_ms" in r:
            total_ms += r["execution_time_ms"]
            count += 1

if count > 0:
    print(f"Average latency across {count} questions: {total_ms / count / 1000:.2f} seconds ({total_ms / count:.0f} ms)")
else:
    print("No latency data found.")
