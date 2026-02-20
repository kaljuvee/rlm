"""
01 - Quickstart: Basic RLM Usage

Demonstrates the simplest RLM call. The model gets a REPL environment where it
can write and execute Python code to produce its answer, instead of generating
text directly. This is the core idea behind RLMs: the model reasons through
code execution rather than pure token generation.

Usage:
    python examples/01_quickstart.py
    python examples/01_quickstart.py --provider anthropic
    python examples/01_quickstart.py --provider gemini --model gemini-2.5-flash
"""

from rlm import RLM

from provider_config import get_backend_kwargs, get_provider_args

args = get_provider_args()
print(f"Using provider: {args.provider}, model: {args.model}")

rlm = RLM(
    backend=args.provider,
    backend_kwargs=get_backend_kwargs(args.provider, args.model),
    verbose=True,
)

# The model will write Python code in its REPL to compute this,
# rather than trying to generate the answer from memory.
result = rlm.completion(
    "Compute the first 20 Fibonacci numbers and return them as a list."
)

print("\n--- Result ---")
print(result.response)
print(f"\n--- Usage ---")
for model, summary in result.usage_summary.model_usage_summaries.items():
    print(f"  Model: {model}")
    print(f"  Total calls: {summary.total_calls}")
    print(f"  Input tokens: {summary.total_input_tokens}")
    print(f"  Output tokens: {summary.total_output_tokens}")
print(f"Execution time: {result.execution_time:.1f}s")
