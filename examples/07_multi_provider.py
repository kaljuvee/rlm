"""
07 - Multi-Provider: Same Task Across Different LLMs

Demonstrates RLM's provider-agnostic design by running the same task across
multiple configured providers and comparing results. This shows that RLMs work
as an inference wrapper on top of any LLM backend.

Usage:
    python examples/07_multi_provider.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from rlm import RLM

load_dotenv()

# Detect which providers have API keys configured
AVAILABLE_PROVIDERS = []
KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

MODEL_MAP = {
    "openai": os.getenv("RLM_OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("RLM_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "gemini": os.getenv("RLM_GEMINI_MODEL", "gemini-2.5-flash"),
}

for provider, env_var in KEY_MAP.items():
    if os.getenv(env_var):
        AVAILABLE_PROVIDERS.append(provider)

if len(AVAILABLE_PROVIDERS) < 2:
    print("This example requires at least 2 providers configured in .env")
    print(f"Found: {AVAILABLE_PROVIDERS or 'none'}")
    print(f"Configure API keys for: {list(KEY_MAP.values())}")
    sys.exit(1)

print(f"Running with providers: {AVAILABLE_PROVIDERS}\n")

TASK = "What is the sum of all prime numbers less than 100? Show your work."

results = {}
for provider in AVAILABLE_PROVIDERS:
    print(f"\n{'='*60}")
    print(f"Provider: {provider} (model: {MODEL_MAP[provider]})")
    print(f"{'='*60}")

    rlm = RLM(
        backend=provider,
        backend_kwargs={"model_name": MODEL_MAP[provider]},
        verbose=True,
    )

    start = time.time()
    result = rlm.completion(TASK)
    elapsed = time.time() - start

    results[provider] = {
        "response": result.response,
        "time": elapsed,
        "tokens": result.usage.total_tokens,
        "cost": result.usage.total_cost,
    }

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
for provider, data in results.items():
    print(f"\n--- {provider} ---")
    print(f"  Time:   {data['time']:.1f}s")
    print(f"  Tokens: {data['tokens']}")
    print(f"  Cost:   ${data['cost']:.4f}")
    print(f"  Answer: {data['response'][:200]}...")
