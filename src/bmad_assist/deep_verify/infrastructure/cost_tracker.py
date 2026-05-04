"""Cost tracking for LLM API calls.

This module provides cost estimation and tracking for LLM usage across
different models and verification methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bmad_assist.deep_verify.infrastructure.types import (
    CostSummary,
    MethodCost,
    ModelCost,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model Pricing
# =============================================================================

# Pricing per 1M tokens (input, output) in USD
# Source: https://www.anthropic.com/pricing, https://openai.com/api/pricing/
# Last updated: 2026-04-24
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_price_per_1m, output_price_per_1m)
    # Anthropic Claude models
    "claude-3-haiku-20240307": (0.25, 1.25),  # Claude 3 Haiku
    "claude-3-sonnet-20240229": (3.00, 15.00),  # Claude 3 Sonnet
    "claude-3-opus-20240229": (15.00, 75.00),  # Claude 3 Opus
    "haiku": (0.25, 1.25),  # Alias for Claude 3 Haiku
    "sonnet": (3.00, 15.00),  # Alias for Claude 3 Sonnet
    "opus": (15.00, 75.00),  # Alias for Claude 3 Opus
    # OpenAI models
    "gpt-5.5": (5.00, 30.00),  # GPT-5.5
    "gpt-5.5-pro": (15.00, 120.00),  # GPT-5.5 pro
    "gpt-5.4-mini": (0.20, 1.60),  # GPT-5.4 mini
    "gpt-4o": (5.00, 15.00),  # GPT-4o
    "gpt-4o-mini": (0.15, 0.60),  # GPT-4o Mini
    "gpt-4-turbo": (10.00, 30.00),  # GPT-4 Turbo
    "gpt-4": (30.00, 60.00),  # GPT-4
    "gpt-3.5-turbo": (0.50, 1.50),  # GPT-3.5 Turbo
    # Z.ai models
    "zai-coding-plan/glm-4.5": (1.00, 3.00),  # GLM 4.5
    "zai-coding-plan/glm-4.7": (1.00, 3.00),  # GLM 4.7
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_INPUT_PRICE = 1.0
DEFAULT_OUTPUT_PRICE = 3.0


def get_model_pricing(model: str) -> tuple[float, float]:
    """Get pricing for a model.

    Args:
        model: Model identifier.

    Returns:
        Tuple of (input_price_per_1m, output_price_per_1m) in USD.
        Returns default pricing if model is unknown.

    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        logger.warning(
            "Unknown model '%s', using default pricing ($%.2f/$%.2f per 1M tokens)",
            model,
            DEFAULT_INPUT_PRICE,
            DEFAULT_OUTPUT_PRICE,
        )
        return (DEFAULT_INPUT_PRICE, DEFAULT_OUTPUT_PRICE)
    return pricing


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate estimated cost in USD.

    Pricing is per 1M tokens. Returns cost in dollars.

    Example:
        >>> # Haiku: $0.25 per 1M input, $1.25 per 1M output
        >>> # 1000 input + 500 output tokens
        >>> calculate_cost(1000, 500, "haiku")
        0.000875

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model identifier.

    Returns:
        Estimated cost in USD.

    """
    input_price, output_price = get_model_pricing(model)

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + output_cost


def estimate_tokens(text: str) -> int:
    """Rough token estimation.

    Rule of thumb: ~4 characters per token for English text.
    This is an approximation - actual tokenization varies by model.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.

    """
    if not text:
        return 0
    return len(text) // 4


# =============================================================================
# Cost Tracker
# =============================================================================


@dataclass
class CostTrackingConfig:
    """Configuration for cost tracking.

    Attributes:
        enabled: Whether cost tracking is enabled.
        track_by_method: Whether to track costs per method.
        track_by_model: Whether to track costs per model.

    """

    enabled: bool = True
    track_by_method: bool = True
    track_by_model: bool = True


class CostTracker:
    """Tracker for LLM costs across verification runs.

    This class accumulates cost information across multiple LLM calls,
    organizing costs by model and by verification method.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record_call("haiku", 1000, 500, "#153")
        >>> summary = tracker.get_summary()
        >>> print(f"Total cost: ${summary.estimated_cost_usd:.4f}")

    """

    def __init__(self, config: CostTrackingConfig | None = None):
        """Initialize cost tracker.

        Args:
            config: Cost tracking configuration. Uses defaults if None.

        """
        self.config = config or CostTrackingConfig()

        # Aggregated totals
        self._total_calls: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0

        # Per-model tracking
        self._model_costs: dict[str, ModelCost] = {}

        # Per-method tracking
        self._method_costs: dict[str, MethodCost] = {}

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        method_id: str | None = None,
    ) -> float:
        """Record an LLM call and update cost tracking.

        Args:
            model: Model identifier used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            method_id: Optional method identifier for attribution.

        Returns:
            Cost of this call in USD.

        """
        if not self.config.enabled:
            return 0.0

        cost = calculate_cost(input_tokens, output_tokens, model)

        # Update totals
        self._total_calls += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost

        # Update per-model costs
        if self.config.track_by_model:
            if model not in self._model_costs:
                self._model_costs[model] = ModelCost()

            current_model = self._model_costs[model]
            self._model_costs[model] = ModelCost(
                calls=current_model.calls + 1,
                input_tokens=current_model.input_tokens + input_tokens,
                output_tokens=current_model.output_tokens + output_tokens,
                estimated_cost_usd=current_model.estimated_cost_usd + cost,
            )

        # Update per-method costs
        if self.config.track_by_method and method_id:
            if method_id not in self._method_costs:
                self._method_costs[method_id] = MethodCost()

            current_method = self._method_costs[method_id]
            self._method_costs[method_id] = MethodCost(
                calls=current_method.calls + 1,
                input_tokens=current_method.input_tokens + input_tokens,
                output_tokens=current_method.output_tokens + output_tokens,
                estimated_cost_usd=current_method.estimated_cost_usd + cost,
            )

        logger.debug(
            "Cost tracked: model=%s, method=%s, tokens=%d+%d, cost=$%.6f",
            model,
            method_id or "unknown",
            input_tokens,
            output_tokens,
            cost,
        )

        return cost

    def get_summary(self) -> CostSummary:
        """Get cost summary for all recorded calls.

        Returns:
            CostSummary with aggregated cost information.

        """
        return CostSummary(
            total_calls=self._total_calls,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_tokens=self._total_input_tokens + self._total_output_tokens,
            estimated_cost_usd=self._total_cost,
            by_model=dict(self._model_costs),
            by_method=dict(self._method_costs),
        )

    def get_model_cost(self, model: str) -> ModelCost | None:
        """Get cost for a specific model.

        Args:
            model: Model identifier.

        Returns:
            ModelCost if model has recorded costs, None otherwise.

        """
        return self._model_costs.get(model)

    def get_method_cost(self, method_id: str) -> MethodCost | None:
        """Get cost for a specific method.

        Args:
            method_id: Method identifier.

        Returns:
            MethodCost if method has recorded costs, None otherwise.

        """
        return self._method_costs.get(method_id)

    def reset(self) -> None:
        """Reset all cost tracking data."""
        self._total_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._model_costs.clear()
        self._method_costs.clear()

        logger.debug("Cost tracker reset")

    def to_dict(self) -> dict[str, Any]:
        """Convert tracker state to dictionary."""
        from bmad_assist.deep_verify.infrastructure.types import serialize_cost_summary

        return serialize_cost_summary(self.get_summary())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostTracker:
        """Create tracker from dictionary."""
        from bmad_assist.deep_verify.infrastructure.types import deserialize_cost_summary

        tracker = cls()
        summary = deserialize_cost_summary(data)

        tracker._total_calls = summary.total_calls
        tracker._total_input_tokens = summary.total_input_tokens
        tracker._total_output_tokens = summary.total_output_tokens
        tracker._total_cost = summary.estimated_cost_usd
        tracker._model_costs = dict(summary.by_model)
        tracker._method_costs = dict(summary.by_method)

        return tracker


# =============================================================================
# No-Op Tracker
# =============================================================================


class NoOpCostTracker:
    """No-op cost tracker that doesn't track anything.

    Use this when cost tracking is disabled but you still need
    the same interface.
    """

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        method_id: str | None = None,
    ) -> float:
        """No-op, returns 0.0."""
        return 0.0

    def get_summary(self) -> CostSummary:
        """Returns empty cost summary."""
        return CostSummary()

    def get_model_cost(self, model: str) -> ModelCost | None:
        """Returns None."""
        return None

    def get_method_cost(self, method_id: str) -> MethodCost | None:
        """Returns None."""
        return None

    def reset(self) -> None:
        """No-op."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Returns empty dict."""
        return {}


# =============================================================================
# Factory Function
# =============================================================================


def create_cost_tracker(enabled: bool = True) -> CostTracker | NoOpCostTracker:
    """Create appropriate cost tracker based on configuration.

    Args:
        enabled: Whether cost tracking is enabled.

    Returns:
        CostTracker if enabled, NoOpCostTracker otherwise.

    Example:
        >>> tracker = create_cost_tracker(enabled=True)
        >>> tracker = create_cost_tracker(enabled=False)  # No-op

    """
    if enabled:
        return CostTracker()
    return NoOpCostTracker()
