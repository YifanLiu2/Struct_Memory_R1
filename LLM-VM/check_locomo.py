import json
import glob
from pathlib import Path

locomo_dir = Path(r"c:\Users\Liam\Desktop\Struct_Memory_R1\LLM-VM\experiment\locomo")
all_convs = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]

for conv in all_convs:
    dirs = list(locomo_dir.glob(f"{conv}_*"))
    if not dirs:
        print(f"{conv}: NO FOLDER")
        continue
    latest = sorted(dirs)[-1]
    res_path = latest / "eval_results.json"
    if res_path.exists():
        try:
            with open(res_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                summary = data.get("summary", {})
                overall = summary.get("overall", {})
                errors = overall.get("error", 0)
                total = summary.get("total_questions", 0)
                print(f"{conv}: {total} Qs, {errors} Errors (in {latest.name})")
        except Exception as e:
            print(f"{conv}: JSON Error {e}")
    else:
        print(f"{conv}: NO JSON")
