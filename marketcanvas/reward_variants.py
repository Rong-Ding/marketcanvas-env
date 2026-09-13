"""Offline reward ablation; the live environment retains its original terminal reward.

Only the essential-validity cap changes. This is graded for contrast/alignment,
not globally continuous: content, visibility, and spacing remain discrete checks.
Success is always measured by the original task contract, independently of score.
"""
from .models import State
from .rewards import evaluate, weighted_quality

BASELINE_VERSION = "capped_v1"
ALTERNATIVE_VERSION = "uncapped_v1"
HIERARCHY_VERSION = "hierarchy_v1"


def evaluate_uncapped(state: State):
    """Score a scene without the validity cap, preserving all diagnostic checks."""
    result = evaluate(state)
    result["score"] = round(2 * weighted_quality(result["components"]) - 1, 6)
    result["reward_variant"] = ALTERNATIVE_VERSION
    return result


def compare_rewards(state: State):
    """Read-only paired measurement of exactly the same scene under both rules."""
    baseline = evaluate(state)
    alternative = round(2 * weighted_quality(baseline["components"]) - 1, 6)
    return {
        BASELINE_VERSION: baseline["score"],
        ALTERNATIVE_VERSION: alternative,
        "success": baseline["success"],
        "essential_constraints_pass": baseline["essential_constraints_pass"],
        "components": baseline["components"],
        "elements": baseline["elements"],
        "issues": baseline["issues"],
    }


def evaluate_hierarchy(state: State, *, style_weight=.15, target_ratio=1.5):
    """Exploratory headline-led typography prior, retaining the validity cap.

    The headline/CTA font-size ratio is only one feature of visual hierarchy.
    1.5 and .15 are declared hypotheses, not literature-derived optimum values.
    Shrinking the CTA can raise this metric; decoration is invisible to it.
    Neither task success nor the live environment's reward is redefined.
    """
    if not 0 <= style_weight <= 1 or not 1 <= target_ratio <= 4:
        raise ValueError("style_weight must be 0..1; target_ratio must be 1..4")
    result = evaluate(state)
    headline = next((e for e in state.elements if e.role == "headline" and e.type == "text"), None)
    cta = next((e for e in state.elements if e.role == "cta" and e.type == "shape"), None)
    hierarchy = min(1.0, headline.font_size / (target_ratio * cta.font_size)) if headline and cta else 0.0
    quality = (1-style_weight) * weighted_quality(result["components"]) + style_weight * hierarchy
    if not result["essential_constraints_pass"]:
        quality = min(quality, .49)
    result.update(score=round(2*quality-1, 6), reward_variant=HIERARCHY_VERSION,
                  aesthetic_components={"headline_size_hierarchy": hierarchy},
                  aesthetic_config={"style_weight": style_weight, "target_ratio": target_ratio})
    return result
