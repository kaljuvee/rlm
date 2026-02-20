# Recursive Language Models (RLMs) - Examples

A collection of examples demonstrating [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601) using the [`rlms`](https://github.com/alexzhang13/rlm) library by Zhang, Krasta and Khattab (MIT OASYS Lab).

## What are RLMs?

Traditional LLMs suffer from **context rot** -- their reasoning degrades as context windows grow. RLMs solve this by treating the LLM as a **programmer** that interacts with data through a REPL (Read-Eval-Print Loop) environment rather than reading everything in one giant prompt.

Key concepts:
- **Context as Variables**: Text is stored as queryable variables, not dumped into the prompt
- **Tool-Mediated Access**: The model writes Python code (grep, regex, pandas) to inspect data
- **Recursion**: Complex tasks spawn child LMs (`rlm_query()`) that analyze data slices independently
- **Code Execution**: Computations are done in a real Python REPL, eliminating hallucinated math

For a detailed introduction, see [Quick Introduction to Recursive Language Models](https://www.linkedin.com/pulse/quick-introduction-recursive-language-models-rlms-julian-kaljuvee-7dabf).

## Setup

```bash
# Clone the repo
git clone https://github.com/kaljuvee/rlm.git
cd rlm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.sample .env
# Edit .env with your API keys
```

## Provider Support

These examples are **LLM-agnostic** and support multiple providers via the `--provider` flag:

| Provider | Backend Key | Required ENV Variable |
|----------|-------------|----------------------|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini` | `GOOGLE_API_KEY` |
| Azure OpenAI | `azure_openai` | `AZURE_OPENAI_API_KEY` + endpoint config |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Portkey | `portkey` | `PORTKEY_API_KEY` |
| LiteLLM | `litellm` | Uses underlying provider keys |

## Examples

All examples accept `--provider` and `--model` flags. Run from the project root:

### 01 - Quickstart
Basic RLM usage -- the model computes Fibonacci numbers via code execution.
```bash
python examples/01_quickstart.py
python examples/01_quickstart.py --provider anthropic
```

### 02 - Context as Variable
The model searches through contract documents programmatically instead of reading them in the prompt window -- demonstrating how RLMs avoid context rot.
```bash
python examples/02_context_as_variable.py
```

### 03 - Recursive Decomposition
The root LM spawns child LMs via `rlm_query()` to independently benchmark sorting algorithms, then synthesizes results -- showcasing the recursive architecture.
```bash
python examples/03_recursive_decomposition.py
```

### 04 - Data Analysis
Parses server logs with real Python code execution. No hallucinated statistics -- all numbers come from actual computation.
```bash
python examples/04_data_analysis.py
```

### 05 - Custom Tools
Injects domain-specific Python functions (financial calculators) into the REPL, extending the model's capabilities.
```bash
python examples/05_custom_tools.py
```

### 06 - Context Compaction
Demonstrates automatic context summarization during long iterative tasks, allowing the model to work through complex problems without hitting token limits.
```bash
python examples/06_compaction.py
```

### 07 - Multi-Provider Comparison
Runs the same task across all configured providers and compares results, timing, and costs.
```bash
python examples/07_multi_provider.py
```

## Running Tests

```bash
# Offline tests (no API key needed)
pytest tests/ -v -k "not completion"

# Full tests (requires API keys in .env)
pytest tests/ -v
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   RLM Framework                  │
├─────────────┬──────────────┬────────────────────┤
│  Root LM    │  REPL Env    │  Context Store     │
│  (Depth 0)  │  (Python)    │  (Variables)       │
│             │              │                    │
│  Reasons &  │  Executes    │  contract_data     │
│  writes     │  real Python │  server_logs       │
│  code       │  code        │  search results    │
├─────────────┴──────────────┴────────────────────┤
│              Recursive Calls                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Child LM │  │ Child LM │  │ Child LM │      │
│  │ (Depth 1)│  │ (Depth 1)│  │ (Depth 1)│      │
│  │ Subtask A│  │ Subtask B│  │ Subtask C│      │
│  └──────────┘  └──────────┘  └──────────┘      │
├─────────────────────────────────────────────────┤
│           LLM Backends (Pluggable)               │
│  OpenAI │ Anthropic │ Gemini │ Azure │ ...      │
└─────────────────────────────────────────────────┘
```

## References

- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601) - Zhang, Krasta, Khattab (MIT)
- [RLM Library](https://github.com/alexzhang13/rlm) - Official implementation
- [Quick Introduction to RLMs](https://www.linkedin.com/pulse/quick-introduction-recursive-language-models-rlms-julian-kaljuvee-7dabf) - Overview article
