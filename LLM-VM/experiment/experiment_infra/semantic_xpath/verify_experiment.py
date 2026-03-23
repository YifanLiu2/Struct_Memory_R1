import argparse
import yaml
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class ExperimentVerifier:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        self.experiment_name = self.config.get("name", "experiment")
        self.output_base_dir = Path("experiment/experiment_result/semantic_xpath")
        self.result_dir = self._find_latest_result_dir()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _find_latest_result_dir(self) -> Optional[Path]:
        """Find the latest result directory for this experiment."""
        candidates = list(self.output_base_dir.glob(f"{self.experiment_name}*"))
        if not candidates:
            return None
        # Sort by modification time, newest first
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    def _load_json_trace(self, trace_path: Path) -> Dict[str, Any]:
        with open(trace_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _find_trace_file(self, query_dir: Path, operation: str) -> Optional[Path]:
        """Find the relevant trace file in the query directory."""
        if not query_dir.exists():
            return None
        
        traces_dir = query_dir / "reasoning_traces"
        if not traces_dir.exists():
            return None

        # Look for crud_{operation}_*.json
        for file in traces_dir.glob(f"crud_{operation.lower()}_*.json"):
            return file
        
        return None

    def _extract_paths_from_trace(self, trace_data: Dict[str, Any], operation: str) -> List[str]:
        """Extract the result paths based on operation type."""
        data = trace_data.get("operation_data", {})
        
        if operation == "READ":
            return [node.get("tree_path") for node in data.get("selected_nodes", [])]
        elif operation == "CREATE":
            return [data.get("created_path")] if data.get("created_path") else []
        elif operation == "UPDATE":
            return data.get("updated_paths", [])
        elif operation == "DELETE":
            return data.get("deleted_paths", [])
        
        return []

    def _normalize_path(self, path: str) -> str:
        """Normalize path string for comparison."""
        if not path:
            return ""
        
        # Replace / with > for consistency
        path = path.replace("/", " > ")
        
        # Split into parts
        parts = [p.strip() for p in path.split(">")]
        
        # Remove empty parts
        parts = [p for p in parts if p]
        
        # Remove Root if present at start
        if parts and parts[0] == "Root":
            parts = parts[1:]
            
        return " > ".join(parts)

    def _detect_operation_from_trace_dir(self, query_dir: Path) -> Optional[str]:
        """Detect the operation type from available trace files in a query directory."""
        traces_dir = query_dir / "reasoning_traces"
        if not traces_dir.exists():
            return None
        
        for op in ["read", "create", "update", "delete"]:
            if list(traces_dir.glob(f"crud_{op}_*.json")):
                return op.upper()
        return None

    def verify(self):
        if not self.result_dir:
            logger.error(f"No result directory found for experiment : {self.experiment_name}")
            return

        logger.info(f"Verifying results in: {self.result_dir}")
        logger.info("=" * 60)

        total_queries = 0
        passed_queries = 0
        failed_queries = 0

        # Map legacy query group names to operation types
        query_type_map = {
            "read_queries": "READ",
            "create_queries": "CREATE",
            "update_queries": "UPDATE",
            "delete_queries": "DELETE"
        }

        # Iterate through query groups defined in yaml
        for group in self.config.get("queries", []):
            group_name = group.get("name")
            fixed_operation = query_type_map.get(group_name)

            logger.info(f"\nChecking {group_name}...")
            
            group_dir = self.result_dir / group_name
            if not group_dir.exists():
                logger.warning(f"Directory not found: {group_dir}")
                continue

            queries = group.get("queries", [])
            # Skip commented-out queries (strings starting with #)
            active_queries = [q for q in queries if isinstance(q, str) and not q.strip().startswith("#")]
            ground_truth = group.get("ground_truth", [])

            if len(active_queries) != len(ground_truth):
                logger.warning(f"Mismatch in queries ({len(active_queries)}) vs ground_truth ({len(ground_truth)}) count for {group_name}")

            # Get sorted query subdirectories to match order
            # Subdirs are named like "01_Query_...", "02_Query_..."
            query_subdirs = sorted([d for d in group_dir.iterdir() if d.is_dir()])

            for idx, (query_text, expected_paths) in enumerate(zip(active_queries, ground_truth)):
                total_queries += 1
                
                if idx >= len(query_subdirs):
                    logger.error(f"  [MISSING] Query {idx+1}: {query_text[:50]}... (No result directory)")
                    failed_queries += 1
                    continue

                query_subdir = query_subdirs[idx]
                
                # Determine operation: use fixed mapping if available, otherwise detect from traces
                operation = fixed_operation or self._detect_operation_from_trace_dir(query_subdir)
                
                if not operation:
                    logger.error(f"  [ERROR] Query {idx+1}: {query_text[:50]}... (Cannot determine operation)")
                    failed_queries += 1
                    continue

                trace_file = self._find_trace_file(query_subdir, operation)

                if not trace_file:
                    logger.error(f"  [MISSING] Query {idx+1}: {query_text[:50]}... (No trace file)")
                    failed_queries += 1
                    continue

                try:
                    trace_data = self._load_json_trace(trace_file)
                    actual_paths = self._extract_paths_from_trace(trace_data, operation)
                except Exception as e:
                    logger.error(f"  [ERROR] Query {idx+1}: Failed to parse trace - {e}")
                    failed_queries += 1
                    continue

                # Prepare expected paths
                target_paths = expected_paths
                is_any_match = False

                if isinstance(expected_paths, dict) and "any" in expected_paths:
                    is_any_match = True
                    target_paths = expected_paths["any"]

                # Compare paths
                clean_expected = sorted([self._normalize_path(p) for p in target_paths if p])
                clean_actual = sorted([self._normalize_path(p) for p in actual_paths if p])
                
                passed = False
                
                if is_any_match:
                    # Flexible match: Check if ANY of the actual results are in the expected set
                    if clean_actual and not set(clean_actual).isdisjoint(set(clean_expected)):
                        passed = True
                else:
                    # Strict match: Exact equality (ignoring order)
                    if clean_expected == clean_actual:
                        passed = True
                    elif set(clean_expected) == set(clean_actual):
                        passed = True

                if passed:
                    logger.info(f"  [PASS] Query {idx+1}: {query_text[:60]}")
                    passed_queries += 1
                else:
                    logger.error(f"  [FAIL] Query {idx+1}: {query_text}")
                    logger.error(f"     Expected: {clean_expected} {'(ANY)' if is_any_match else ''}")
                    logger.error(f"     Actual:   {clean_actual}")
                    failed_queries += 1

        logger.info("=" * 60)
        logger.info(f"Verification Summary")
        logger.info(f"Total Queries: {total_queries}")
        logger.info(f"Passed: {passed_queries}")
        logger.info(f"Failed: {failed_queries}")
        
        if total_queries > 0:
            pass_rate = (passed_queries / total_queries) * 100
            logger.info(f"Pass Rate: {pass_rate:.1f}%")
        
        if failed_queries > 0:
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Verify experiment results against ground truth.")
    parser.add_argument("--config", required=True, help="Path to experiment config yaml with ground truth")
    args = parser.parse_args()

    verifier = ExperimentVerifier(args.config)
    verifier.verify()

if __name__ == "__main__":
    main()
