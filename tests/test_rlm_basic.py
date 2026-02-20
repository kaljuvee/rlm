"""
Basic integration tests for RLM examples.

These tests verify the RLM library can be imported and configured correctly,
and run a simple end-to-end completion if an API key is available.

Run:
    pytest tests/ -v
    pytest tests/ -v -k "test_completion" --provider openai  # with live API
"""

import os
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()

# Add examples dir to path for provider_config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_import_rlm():
    """Verify the rlm package is installed and importable."""
    from rlm import RLM

    assert RLM is not None


def test_import_exceptions():
    """Verify RLM exception types are importable."""
    from rlm import (
        BudgetExceededError,
        ErrorThresholdExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
    )

    assert BudgetExceededError is not None
    assert TimeoutExceededError is not None
    assert TokenLimitExceededError is not None
    assert ErrorThresholdExceededError is not None


def test_provider_config():
    """Verify provider_config module loads and parses correctly."""
    from provider_config import DEFAULT_MODELS, PROVIDERS, get_backend_kwargs

    assert "openai" in PROVIDERS
    assert "anthropic" in PROVIDERS
    assert "gemini" in PROVIDERS
    assert "openai" in DEFAULT_MODELS

    kwargs = get_backend_kwargs("openai", "gpt-4o-mini")
    assert kwargs["model_name"] == "gpt-4o-mini"


def test_azure_backend_kwargs():
    """Verify Azure backend kwargs include Azure-specific fields."""
    from provider_config import get_backend_kwargs

    kwargs = get_backend_kwargs("azure_openai", "gpt-4o-mini")
    assert "azure_endpoint" in kwargs
    assert "azure_deployment" in kwargs
    assert "api_version" in kwargs


def test_sample_data_exists():
    """Verify sample data files are present."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    assert os.path.isfile(os.path.join(data_dir, "sample_contracts.txt"))
    assert os.path.isfile(os.path.join(data_dir, "server_logs.txt"))


def test_rlm_instantiation():
    """Verify RLM can be instantiated (without making API calls)."""
    from rlm import RLM

    # This should work even without an API key -- it only fails on .completion()
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini"},
    )
    assert rlm is not None


# Live API tests -- only run when API key is available
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_completion_openai():
    """End-to-end test: run a simple RLM completion with OpenAI."""
    from rlm import RLM

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": os.getenv("RLM_OPENAI_MODEL", "gpt-4o-mini")},
        max_iterations=5,
    )

    result = rlm.completion(
        "What is 2 + 2? Compute it in Python using FINAL_VAR() to return the answer."
    )
    assert result is not None
    assert result.response is not None
    # The model should complete without error; exact response depends on model behavior
    assert len(str(result.response)) > 0


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_completion_anthropic():
    """End-to-end test: run a simple RLM completion with Anthropic."""
    from rlm import RLM

    rlm = RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": os.getenv("RLM_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        },
        max_iterations=5,
    )

    result = rlm.completion(
        "What is 3 * 7? Compute it in Python using FINAL_VAR() to return the answer."
    )
    assert result is not None
    assert result.response is not None
    assert len(str(result.response)) > 0


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
def test_completion_gemini():
    """End-to-end test: run a simple RLM completion with Gemini."""
    from rlm import RLM

    rlm = RLM(
        backend="gemini",
        backend_kwargs={
            "model_name": os.getenv("RLM_GEMINI_MODEL", "gemini-2.5-flash")
        },
        max_iterations=5,
    )

    result = rlm.completion(
        "What is 10 ** 3? Compute it in Python using FINAL_VAR() to return the answer."
    )
    assert result is not None
    assert result.response is not None
    assert len(str(result.response)) > 0
