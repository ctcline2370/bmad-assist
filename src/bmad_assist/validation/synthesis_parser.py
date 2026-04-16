"""Synthesis metrics parser for extracting structured output.

Story 13.6: Synthesizer Schema Integration

This module provides:
- SynthesisMetrics dataclass for parsed metrics
- extract_synthesis_metrics() for marker-based extraction

The synthesis workflow outputs structured JSON between markers:
<!-- METRICS_JSON_START --> and <!-- METRICS_JSON_END -->

Extraction is graceful - failures log warnings and return None rather than
raising exceptions, to avoid blocking synthesis phase completion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from bmad_assist.benchmarking.schema import ConsensusData, QualitySignals

__all__ = [
    "SynthesisMetrics",
    "extract_synthesis_metrics",
]

logger = logging.getLogger(__name__)

# Marker strings for JSON extraction
_METRICS_START = "<!-- METRICS_JSON_START -->"
_METRICS_END = "<!-- METRICS_JSON_END -->"


@dataclass(frozen=True)
class SynthesisMetrics:
    """Metrics extracted from synthesis output.

    Both fields may be None if extraction failed for that section.
    Partial extraction is allowed - quality may succeed while consensus fails.
    """

    quality: QualitySignals | None
    consensus: ConsensusData | None


def extract_synthesis_metrics(raw_output: str) -> SynthesisMetrics | None:
    """Extract structured metrics from synthesis output.

    Uses marker-based extraction to find JSON between:
    <!-- METRICS_JSON_START --> and <!-- METRICS_JSON_END -->

    Extraction is graceful:
    - Missing markers -> log warning, return None
    - Invalid JSON -> log warning, return None
    - Schema validation failure for one section -> log warning, return partial
    - Both sections fail -> return None

    Args:
        raw_output: Raw synthesis LLM output.

    Returns:
        SynthesisMetrics with quality and/or consensus, or None if extraction
        completely fails (no valid sections).

    """
    # Find markers
    start_idx = raw_output.find(_METRICS_START)
    end_idx = raw_output.find(_METRICS_END)

    if start_idx == -1 or end_idx == -1:
        # METRICS_JSON markers are not yet emitted by synthesis prompts — this is expected.
        # Downgraded from WARNING to DEBUG to avoid log noise until prompt integration is added.
        logger.debug(
            "METRICS_JSON markers not found in synthesis output (len=%d), skipping metrics extraction",
            len(raw_output),
        )
        return None

    # Extract JSON between markers
    json_str = raw_output[start_idx + len(_METRICS_START) : end_idx].strip()

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        excerpt = raw_output[:500]
        logger.warning(
            "Invalid JSON in synthesis metrics: %s (output len=%d): %s...",
            e,
            len(raw_output),
            excerpt,
        )
        return None

    # Validate quality section
    quality: QualitySignals | None = None
    excerpt = raw_output[:500]
    if "quality" in data:
        try:
            quality = QualitySignals.model_validate(data["quality"])
        except ValidationError as e:
            logger.warning(
                "Quality schema validation failed: %s (output len=%d): %s...",
                e,
                len(raw_output),
                excerpt,
            )

    # Validate consensus section
    consensus: ConsensusData | None = None
    if "consensus" in data:
        try:
            consensus = ConsensusData.model_validate(data["consensus"])
        except ValidationError as e:
            logger.warning(
                "Consensus schema validation failed: %s (output len=%d): %s...",
                e,
                len(raw_output),
                excerpt,
            )

    # If both failed, return None
    if quality is None and consensus is None:
        logger.warning("Both quality and consensus schema validation failed for synthesis output")
        return None

    return SynthesisMetrics(quality=quality, consensus=consensus)
