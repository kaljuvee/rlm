"""
Shared provider configuration for all examples.

Reads from .env to determine which LLM backend to use and configures the RLM
instance accordingly. Supports: openai, anthropic, gemini, azure_openai,
openrouter, portkey, litellm.
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

# Default models per provider
DEFAULT_MODELS = {
    "openai": os.getenv("RLM_OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("RLM_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "gemini": os.getenv("RLM_GEMINI_MODEL", "gemini-2.5-flash"),
    "azure_openai": os.getenv("RLM_AZURE_MODEL", "gpt-4o-mini"),
    "openrouter": os.getenv("RLM_OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    "portkey": os.getenv("RLM_PORTKEY_MODEL", "gpt-4o-mini"),
    "litellm": os.getenv("RLM_LITELLM_MODEL", "gpt-4o-mini"),
}

PROVIDERS = list(DEFAULT_MODELS.keys())


def get_provider_args() -> argparse.Namespace:
    """Parse --provider and --model CLI args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=os.getenv("RLM_DEFAULT_PROVIDER", "openai"),
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (defaults to provider-specific default)",
    )
    args, _ = parser.parse_known_args()
    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]
    return args


def get_backend_kwargs(provider: str, model: str) -> dict:
    """Build backend_kwargs dict for the given provider."""
    kwargs = {"model_name": model}

    if provider == "azure_openai":
        kwargs["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        kwargs["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        kwargs["azure_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        kwargs["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    elif provider == "portkey":
        kwargs["api_key"] = os.getenv("PORTKEY_API_KEY")

    return kwargs
