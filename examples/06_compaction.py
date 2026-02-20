"""
06 - Context Compaction: Handling Long Iterative Tasks

Demonstrates RLM's context compaction feature. When the conversation history
approaches the model's token limit, RLM automatically summarizes prior
iterations to free up space -- allowing the model to work through arbitrarily
long tasks without losing track of progress.

Usage:
    python examples/06_compaction.py
    python examples/06_compaction.py --provider anthropic
"""

from rlm import RLM

from provider_config import get_backend_kwargs, get_provider_args

args = get_provider_args()
print(f"Using provider: {args.provider}, model: {args.model}")

rlm = RLM(
    backend=args.provider,
    backend_kwargs=get_backend_kwargs(args.provider, args.model),
    max_iterations=15,
    compaction=True,
    compaction_threshold_pct=0.5,  # Compact when 50% of context is used
    verbose=True,
)

result = rlm.completion(
    "Implement and test a complete linked list data structure in Python with "
    "the following methods: append, prepend, delete, find, reverse, and __len__. "
    "Write unit tests for each method, run them, and report the results."
)

print("\n--- Result ---")
print(result.response)
