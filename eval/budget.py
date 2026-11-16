"""
eval/budget.py
Token budget enforcement for evaluation runs.
Tracks token consumption using usage statistics from llm.py and raises
BudgetExceeded when cumulative tokens consumed exceed the configured limit.
"""

from typing import Any
import llm


class BudgetExceeded(Exception):
    """Raised when an evaluation run exceeds its allocated token budget."""

    def __init__(
        self,
        used_tokens: int,
        max_tokens: int,
        message: str | None = None,
    ):
        self.used_tokens = used_tokens
        self.max_tokens = max_tokens
        default_msg = (
            f"Token budget exceeded: consumed {used_tokens:,} tokens "
            f"(budget limit: {max_tokens:,} tokens)"
        )
        super().__init__(message or default_msg)


class TokenBudget:
    """
    Monitors token usage across LLM invocations using llm.get_usage_stats().
    """

    def __init__(self, max_tokens: int | None = None):
        """
        Initializes the budget tracker.

        Args:
            max_tokens: Maximum tokens allowed since instantiation.
                        None means unlimited tokens.
        """
        self.max_tokens = max_tokens if max_tokens is not None and max_tokens > 0 else (
            0 if max_tokens == 0 else None
        )
        self._initial_tokens = self._current_total()

    @staticmethod
    def _current_total() -> int:
        stats = llm.get_usage_stats()
        return int(stats.get("total_tokens", 0))

    @property
    def tokens_used(self) -> int:
        """Returns the number of tokens consumed since this budget was initialized."""
        return max(0, self._current_total() - self._initial_tokens)

    def get_tokens_used(self) -> int:
        """Convenience method returning tokens consumed."""
        return self.tokens_used

    def check(self) -> None:
        """
        Checks current token usage against the budget limit.
        Raises:
            BudgetExceeded: If max_tokens is set and tokens_used >= max_tokens.
        """
        if self.max_tokens is not None and self.tokens_used >= self.max_tokens:
            raise BudgetExceeded(
                used_tokens=self.tokens_used,
                max_tokens=self.max_tokens,
            )
