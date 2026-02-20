"""
05 - Custom Tools: Extending the REPL with Domain Functions

Demonstrates injecting custom Python functions into the RLM's REPL environment.
The model can call these tools from within its generated code, enabling
integration with external APIs, databases, or domain-specific logic.

Usage:
    python examples/05_custom_tools.py
    python examples/05_custom_tools.py --provider anthropic
"""

import math

from rlm import RLM

from provider_config import get_backend_kwargs, get_provider_args

args = get_provider_args()
print(f"Using provider: {args.provider}, model: {args.model}")


# Define custom tools that will be available in the REPL
def compound_interest(principal: float, rate: float, years: int, n: int = 12) -> dict:
    """Calculate compound interest with monthly compounding."""
    amount = principal * (1 + rate / n) ** (n * years)
    return {
        "principal": principal,
        "rate": f"{rate*100:.1f}%",
        "years": years,
        "final_amount": round(amount, 2),
        "interest_earned": round(amount - principal, 2),
    }


def loan_payment(principal: float, annual_rate: float, years: int) -> dict:
    """Calculate monthly loan payment using amortization formula."""
    monthly_rate = annual_rate / 12
    n_payments = years * 12
    if monthly_rate == 0:
        payment = principal / n_payments
    else:
        payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / (
            (1 + monthly_rate) ** n_payments - 1
        )
    return {
        "monthly_payment": round(payment, 2),
        "total_paid": round(payment * n_payments, 2),
        "total_interest": round(payment * n_payments - principal, 2),
    }


rlm = RLM(
    backend=args.provider,
    backend_kwargs=get_backend_kwargs(args.provider, args.model),
    custom_tools={
        "compound_interest": {
            "tool": compound_interest,
            "description": (
                "Calculate compound interest. Args: principal (float), "
                "rate (float, e.g. 0.05 for 5%), years (int), n (int, compounding periods/year, default 12)"
            ),
        },
        "loan_payment": {
            "tool": loan_payment,
            "description": (
                "Calculate monthly loan payment. Args: principal (float), "
                "annual_rate (float, e.g. 0.065 for 6.5%), years (int)"
            ),
        },
    },
    verbose=True,
)

result = rlm.completion(
    "A client wants to compare two options:\n"
    "Option A: Invest $100,000 at 7% annual return for 20 years.\n"
    "Option B: Take a $100,000 loan at 6.5% for 20 years, invest the same "
    "$100,000 at 9% for 20 years, and pay off the loan from investment gains.\n\n"
    "Use the available tools (compound_interest, loan_payment) to calculate "
    "both scenarios and recommend which option is better. Show all the numbers."
)

print("\n--- Result ---")
print(result.response)
