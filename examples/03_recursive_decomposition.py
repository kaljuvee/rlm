"""
03 - Recursive Decomposition: Spawning Child LMs

Demonstrates the recursive nature of RLMs. The root LM (depth 0) can spawn
child LMs via `rlm_query()` to handle subtasks independently. Each child gets
its own REPL and context, processes a slice of the problem, and returns a
summary to the parent.

This is how RLMs scale to massive datasets: instead of one model struggling
with everything, the work is recursively divided and conquered.

Usage:
    python examples/03_recursive_decomposition.py
    python examples/03_recursive_decomposition.py --provider anthropic
"""

from rlm import RLM

from provider_config import get_backend_kwargs, get_provider_args

args = get_provider_args()
print(f"Using provider: {args.provider}, model: {args.model}")

rlm = RLM(
    backend=args.provider,
    backend_kwargs=get_backend_kwargs(args.provider, args.model),
    max_depth=2,  # Allow up to 2 levels of recursive calls
    verbose=True,
)

# This task naturally decomposes: the root LM can spawn child LMs to
# research each topic independently, then synthesize the results.
result = rlm.completion(
    "Compare three sorting algorithms: quicksort, mergesort, and heapsort. "
    "For each algorithm, use `rlm_query()` to spawn a child analysis that: "
    "1) Implements the algorithm in Python, "
    "2) Tests it on a random list of 1000 integers, "
    "3) Reports the execution time. "
    "Then synthesize the results into a comparison table."
)

print("\n--- Result ---")
print(result.response)
print(f"\n--- Metadata ---")
if result.metadata:
    print(f"Metadata: {result.metadata}")
print(f"Execution time: {result.execution_time:.1f}s")
