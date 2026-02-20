"""
04 - Data Analysis: Code Execution Over Structured Data

Demonstrates RLMs executing Python code to analyze data accurately. Instead of
the model hallucinating statistics, it writes actual pandas/Python code in the
REPL, executes it, and reports computed results.

This solves the "calculator problem" -- LLMs are bad at arithmetic, but RLMs
offload computation to real code execution.

Usage:
    python examples/04_data_analysis.py
    python examples/04_data_analysis.py --provider anthropic
"""

import os

from rlm import RLM

from provider_config import get_backend_kwargs, get_provider_args

args = get_provider_args()
print(f"Using provider: {args.provider}, model: {args.model}")

# Load server logs
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "server_logs.txt")
with open(data_path) as f:
    logs = f.read()

rlm = RLM(
    backend=args.provider,
    backend_kwargs=get_backend_kwargs(args.provider, args.model),
    verbose=True,
)

result = rlm.completion(
    "The variable `context` contains server monitoring logs. "
    "Write Python code to parse these logs and produce a report with:\n"
    "1. Average CPU, memory, and latency per server\n"
    "2. Which server had the highest peak CPU usage and when\n"
    "3. How many CRITICAL/DEGRADED events occurred per server\n"
    "4. A recommendation for which server needs the most attention and why",
    context=logs,
)

print("\n--- Result ---")
print(result.response)
