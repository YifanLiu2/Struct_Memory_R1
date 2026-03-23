import os
import json
import csv
import glob
from pathlib import Path

BASE_DIR = r"C:\Users\Liam\Desktop\LLM-VM\experiment\experiment_result\semantic_xpath"
MODELS = {
    "nutrition_deberta_small": "DeBERTa-Small",
    "nutrition_deberta_base": "DeBERTa-Base",
    "nutrition_experiment_2": "BART-Large"
}

OUTPUT_CSV = os.path.join(BASE_DIR, "model_compare.csv")

def get_trace_data(model_dir, model_name):
    # Find all execution traces (contain timing and result count)
    # Pattern: model_dir/*/query_dir/reasoning_traces/execution_*.json
    # Actually, let's look for crud_*.json for high level timing, and execution_*.json for result count and xpath timing
    
    results = []
    
    # Walk through query directories
    for root, dirs, files in os.walk(model_dir):
        if "reasoning_traces" in root:
            # We are in a trace folder
            # Find the execution trace
            exec_files = [f for f in files if f.startswith("execution_")]
            crud_files = [f for f in files if f.startswith("crud_")]
            
            if not exec_files or not crud_files:
                continue
                
            # Sort by time to get latest run if multiple
            exec_files.sort(reverse=True)
            crud_files.sort(reverse=True)
            
            exec_path = os.path.join(root, exec_files[0])
            crud_path = os.path.join(root, crud_files[0])
            
            try:
                # Load execution data (XPath time + Nodes)
                with open(exec_path, 'r', encoding='utf-8') as f:
                    exec_data = json.load(f)
                
                # Load Handler data if available (Downstream time)
                # Handler file is usually named read_handler_... or create_handler_... or similar
                # We can find it by matching timestamp or just grabbing the only handler file
                handler_files = [f for f in files if "handler_" in f]
                handler_data = {}
                if handler_files:
                    handler_files.sort(reverse=True)
                    with open(os.path.join(root, handler_files[0]), 'r', encoding='utf-8') as f:
                        handler_data = json.load(f)
                
                # Extract Query Name
                query_folder = Path(root).parent.name
                parts = query_folder.split("_")
                query_id = parts[0] 
                query_desc = "_".join(parts[2:6])
                
                # Metrics
                xpath_time_ms = exec_data.get("execution_time_ms", 0)
                downstream_time_ms = handler_data.get("processing_time_ms", 0)
                total_time_ms = xpath_time_ms + downstream_time_ms # Approx
                
                # Accuracy proxy: Number of result nodes
                # From traversal steps
                nodes_found = 0
                if "traversal_steps" in exec_data and exec_data["traversal_steps"]:
                    last_step = exec_data["traversal_steps"][-1]
                    nodes_found = last_step.get("nodes_after_count", 0)
                
                results.append({
                    "Model": model_name,
                    "Query_ID": query_id,
                    "Query_Name": query_desc,
                    "Total_Time_s": round(total_time_ms / 1000, 2),
                    "XPath_Time_s": round(xpath_time_ms / 1000, 2),
                    "Nodes_Found": nodes_found,
                    "Status": "SUCCESS"
                })
                
            except Exception as e:
                print(f"Error parsing {root}: {e}")
                
    return results

all_data = []
for dir_name, model_name in MODELS.items():
    full_path = os.path.join(BASE_DIR, dir_name)
    if os.path.exists(full_path):
        print(f"Processing {model_name} from {dir_name}...")
        all_data.extend(get_trace_data(full_path, model_name))
    else:
        print(f"Directory not found: {dir_name}")

# Sort by Query ID then Model
all_data.sort(key=lambda x: (x["Query_ID"], x["Model"]))

# Write CSV
keys = ["Model", "Query_ID", "Query_Name", "Total_Time_s", "XPath_Time_s", "Nodes_Found", "Status"]
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(all_data)

print(f"Report saved to {OUTPUT_CSV}")
