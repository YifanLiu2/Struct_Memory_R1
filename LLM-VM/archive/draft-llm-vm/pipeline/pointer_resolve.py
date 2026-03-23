from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_intepreter import resolve_pointer


def process_request(request: str) -> dict:
    """
    Process a user request to find related POIs/restaurants.
    
    Args:
        request: Natural language request
        
    Returns:
        dict with matched entities and details
    """
    return resolve_pointer(request)


def interactive_mode():
    """Run in interactive mode, continuously accepting requests."""
    print("=" * 60)
    print("Toronto Itinerary Pointer Resolver")
    print("=" * 60)
    print("Enter your requests to find related POIs/restaurants.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            request = input("\n> Enter request: ").strip()
            
            if not request:
                continue
            
            if request.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            print("\nProcessing...")
            result = process_request(request)
            
            print("\n--- RESULTS ---")
            print(f"Confidence: {result.get('confidence', 'unknown')}")
            print(f"Matched IDs: {result.get('matched_ids', [])}")
            
            # Show reasoning trace
            trace = result.get("reasoning_trace", {})
            if trace.get("stage1"):
                print(f"\nStage 1 candidates: {trace['stage1'].get('candidate_ids', [])}")
                print(f"Stage 1 reasoning: {trace['stage1'].get('reasoning', 'N/A')}")
            if trace.get("stage2"):
                print(f"\nStage 2 reasoning: {trace['stage2'].get('reasoning', 'N/A')}")
                if trace['stage2'].get('filtered_out'):
                    print(f"Filtered out: {trace['stage2']['filtered_out']}")
            
            if result.get("matched_entities"):
                print("\n--- MATCHED ENTITIES ---")
                for eid, details in result["matched_entities"].items():
                    print(f"\n[{eid}] {details.get('poi_name', 'Unknown')}")
                    print(f"  Day: {details.get('day', 'N/A')}")
                    print(f"  Time: {details.get('time_block', 'N/A')}")
                    print(f"  Description: {details.get('description', 'N/A')}")
            
            print(f"\nTrace saved to: {result.get('trace_folder', 'N/A')}")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    interactive_mode()
