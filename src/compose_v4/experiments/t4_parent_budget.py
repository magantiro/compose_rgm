"""Explicit finite-compute allocation for the approved T4 feedback round."""

from contextlib import contextmanager
from unittest.mock import patch

from compose_v4.experiments.continuation_profile import _ExecutorBudgetStop
from compose_v4.rewrite.kernel import RewriteSystem


class _ParentShareStop(_ExecutorBudgetStop):
    def __init__(self, owner):
        self.owner = owner
        super().__init__("parent executor share exhausted")


class ParentExecutorShare:
    """Count the same public calls as the enclosing round meter, including nesting.

    Only this share's administrative stop is caught. Invalid chemistry, runtime
    errors and the round-wide hard stop retain their existing behavior. This
    scoped instrumentation is restricted to the single-worker preparation lane.
    """

    def __init__(self, limit: int | None):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("parent executor share must be a positive integer")
        self.limit = limit
        self.calls = 0
        self.exhausted = False

    @contextmanager
    def instrument(self):
        original = RewriteSystem.apply

        def apply(system, state, rule, action):
            if self.limit is not None and self.calls >= self.limit:
                raise _ParentShareStop(self)
            self.calls += 1
            return original(system, state, rule, action)

        try:
            with patch.object(RewriteSystem, "apply", apply):
                yield self
        except _ParentShareStop as error:
            if error.owner is not self:
                raise
            self.exhausted = True

    def receipt(self) -> dict:
        return {
            "policy": "equal_parent_executor_shares_v1",
            "limit": self.limit,
            "executor_calls": self.calls,
            "exhausted": self.exhausted,
        }
