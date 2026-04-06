"""Tracks cumulative cost across iterations.

Since we use free OpenRouter models, the only real cost is Vapi assistant
operations (currently $0). Budget tracking is kept as a hook for when
non-free models are configured.
"""
from __future__ import annotations


# Approximate price per 1M tokens (USD) — update when swapping to paid models
_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    # model_id: (input_price, output_price)
    "meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "meta-llama/llama-3.1-8b-instruct:free": (0.0, 0.0),
    # Paid fallbacks for reference
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-opus-4": (15.0, 75.0),
}


class BudgetTracker:
    def __init__(self, budget_usd: float):
        self._budget = budget_usd
        self._cumulative = 0.0
        self._iteration_cost = 0.0

    def add_llm_cost(self, model: str, input_tokens: int, output_tokens: int) -> None:
        in_price, out_price = _PRICE_PER_1M.get(model, (0.0, 0.0))
        cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        self._iteration_cost += cost
        self._cumulative += cost

    def add_vapi_cost(self, cost: float) -> None:
        self._iteration_cost += cost
        self._cumulative += cost

    def end_iteration(self) -> float:
        """Return and reset iteration cost."""
        cost = self._iteration_cost
        self._iteration_cost = 0.0
        return cost

    @property
    def cumulative(self) -> float:
        return self._cumulative

    @property
    def budget(self) -> float:
        return self._budget

    def exhausted(self) -> bool:
        return self._cumulative >= self._budget
