import torch
import time
from client.bart_client import BartNLIClient

print(" Checking Device Support...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Running on CPU! This explains the slowness.")

print("\n--- Initializing Client ---")
start_time = time.time()
client = BartNLIClient()
print(f"Client loaded in {time.time() - start_time:.2f}s")
print(f"Model Device: {client.model.device}")

print("\n--- Running Inference Speed Test ---")
premises = ["A healthy meal with quinoa and vegetables." for _ in range(64)]
predicate = "healthy"

# Warmup
print("Warming up...")
client.get_entailment_score(premises[0], predicate)

# Single item test
print("\nTesting single item inference...")
t0 = time.time()
client.get_entailment_score(premises[0], predicate)
print(f"Single item time: {(time.time() - t0)*1000:.2f} ms")

# Batch 64 test
print("\nTesting Batch-64 inference...")
t0 = time.time()
client.batch_entailment_scores(premises, predicate)
duration = time.time() - t0
print(f"Batch-64 time: {duration:.2f} s")
print(f"Time per item: {(duration/64)*1000:.2f} ms/item")
