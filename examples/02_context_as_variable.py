"""
02 - Context as Variable: Querying Data Without Context Rot

Demonstrates the key RLM insight: instead of dumping large text into the prompt
(which causes "context rot" -- degraded reasoning as context grows), RLMs store
context as a queryable variable that the model searches programmatically.

The model receives the contract data as `context` and writes Python code
(regex, string search, etc.) to extract specific information -- achieving
perfect recall regardless of document size.

Usage:
    python examples/02_context_as_variable.py
    python examples/02_context_as_variable.py --provider anthropic
"""

import os

from rlm import RLM

from provider_config import get_backend_kwargs, get_provider_args

args = get_provider_args()
print(f"Using provider: {args.provider}, model: {args.model}")

# Load sample contracts as context
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_contracts.txt")
with open(data_path) as f:
    contracts = f.read()

rlm = RLM(
    backend=args.provider,
    backend_kwargs=get_backend_kwargs(args.provider, args.model),
    verbose=True,
)

# The model will use the REPL to search through the contracts programmatically
# rather than relying on attention over a large context window.
result = rlm.completion(
    "The variable `context` contains multiple contracts. "
    "Find all contracts with a liability cap of $1,000,000 or more. "
    "For each, report the contract number, parties, and exact liability cap. "
    "Also calculate the total combined liability exposure.",
    context=contracts,
)

print("\n--- Result ---")
print(result.response)
