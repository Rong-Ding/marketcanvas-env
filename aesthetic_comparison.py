"""Explore a typography rule with counterexamples and score-hidden visual review.

This tests discrimination and sensitivity, not validated aesthetic preference.
Run: python aesthetic_comparison.py
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from PIL import Image, ImageDraw

from marketcanvas.rendering import font
from marketcanvas.reward_variants import evaluate_hierarchy
from marketcanvas.scenarios import apply, load_scenario
from reward_comparison import ROOT, write_csv

CASES = {
    "A": ("equal_type_sizes", 28, 28, False),
    "B": ("reference_hierarchy", 42, 28, False),
    "C": ("smaller_cta_shortcut", 28, 16, False),
    "D": ("intermediate_hierarchy", 35, 28, False),
    "E": ("large_headline", 56, 28, False),
    "F": ("competing_decorative_text", 42, 28, True),
}


def scene(case, seed=0):
    name, headline_size, cta_size, distractor = CASES[case]
    env = load_scenario(seed=seed)
    apply(env, {"op": "update_element", "id": "e1", "properties": {"font_size": headline_size}})
    apply(env, {"op": "update_element", "id": "e3", "properties": {"font_size": cta_size}})
    if distractor:
        # Leave required glyphs clear while introducing a competing message.
        apply(env, {"op": "add_element", "element": {
            "type": "text", "role": "decoration", "content": "MEGA BONUS", "font_size": 56,
            "x": 50, "y": 190, "width": 700, "height": 64, "z_index": 1,
        }})
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/aesthetic-comparison"))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows, sensitivity = [], []
    sheet = Image.new("RGB", (1280, 1572), "#EEF2F6")
    for seed in range(12):
        for index, case in enumerate(CASES):
            env = scene(case, seed)
            baseline, augmented = env.current_reward(), evaluate_hierarchy(env.state)
            rows.append({"seed": seed, "panel": case, "condition": CASES[case][0],
                         "capped_v1": baseline["score"], "hierarchy_v1": augmented["score"],
                         "hierarchy": augmented["aesthetic_components"]["headline_size_hierarchy"],
                         "success": baseline["success"]})
            for weight in (.05, .15, .30):
                for ratio in (1.25, 1.5, 2.0):
                    result = evaluate_hierarchy(env.state, style_weight=weight, target_ratio=ratio)
                    sensitivity.append({"seed": seed, "panel": case, "weight": weight,
                                        "target_ratio": ratio, "score": result["score"]})
            if seed == 0:
                env.save_png(out / f"panel_{case}.png")
                (out / f"panel_{case}.json").write_text(json.dumps({
                    "observation": env.observe(), "baseline": baseline, "hierarchy": augmented,
                }, indent=2) + "\n")
                tile = Image.new("RGB", (640, 524), "#EEF2F6")
                tile.paste(Image.fromarray(env.render()).resize((608, 456)), (16, 46))
                ImageDraw.Draw(tile).text((16, 12), f"Design {case}", font=font(22), fill="#172033")
                sheet.paste(tile, ((index % 2)*640, (index // 2)*524))
    sheet.save(out / "review_designs.png")
    write_csv(out / "paired_results.csv", rows)
    write_csv(out / "parameter_sensitivity.csv", sensitivity)
    # A blank template, not fabricated human data. IDs align with panel PNGs.
    write_csv(out / "review_template.csv", [
        {"reviewer_id": "", "pair": pair, "preferred_design": "", "headline_prominence": "",
         "cta_readability": "", "overall_preference": "", "reason": ""}
        for pair in ("A vs B", "A vs D", "B vs C", "B vs E", "B vs F")])
    metadata = {
        "experiment": "headline_hierarchy_ablation_v1", "variants": ["capped_v1", "hierarchy_v1"],
        "style_weight": .15, "target_ratio": 1.5, "seed_count": 12, "paired_scenes": len(rows),
        "parameter_combinations": 9, "sensitivity_rows": len(sensitivity),
        "brief_assumption": "Explore headline-led marketing layouts; this is not an additional task-success requirement.",
        "action_mode": "semantic", "observation_mode": "semantic_json", "policy": "scripted; no LLM",
        "human_ratings_collected": 0, "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("pillow", "numpy", "pydantic")},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "marketcanvas").glob("*.py")) + [Path(__file__).resolve(), ROOT / "reward_comparison.py"]},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = ["# Aesthetic-rule pilot: headline hierarchy", "",
        "**Question:** can one explicit typography preference distinguish valid layouts the original reward treats as identical? "
        "**Result:** yes, for relative headline size. Whether those distinctions improve human preference remains open; "
        "the rule also rewards reducing CTA size and ignores competing decorative text.", "",
        "## Inspect before reading the scores", "",
        "Assume a headline-led campaign. Compare headline prominence, CTA readability and overall preference separately; ties are allowed. "
        "The images omit scores to reduce anchoring, but this is an exploratory review sheet, not a randomized or blinded validation study. "
        "Use the blank `review_template.csv` to record judgments. No human ratings have been collected.", "",
        "![Six layouts without score labels](review_designs.png)", "",
        "## Rule and rationale", "",
        "Let `H = min(1, headline_font_size / (1.5 * CTA_font_size))`. "
        "The alternative uses `q_new = .85 * q_original_before_cap + .15 * H`, then reapplies the original essential-validity cap and maps quality to [-1,1]. "
        "The original task-success flag stays unchanged. The target ratio 1.5 and weight .15 are explicit hypotheses, not universal design laws. "
        "A zero style weight exactly recovers the original reward. This study keeps the cap; the contrast study removes it. "
        "Separating these changes avoids confusing their effects.", "",
        "There is precedent for operationalizing design principles mathematically: "
        "[Lok, Feiner and Ngai (2004)](https://www.cs.columbia.edu/~lok/papers/balance.pdf) develop a computable visual-balance measure. "
        "[Yang et al. (2024)](https://arxiv.org/abs/2403.18183) investigate document-design manipulations including font-size contrast. "
        "Neither source validates our specific ratio or establishes marketing effectiveness for this simulator.", "",
        "## Measured scores (default task)", "",
        "| Panel | Manipulation | Original | With hierarchy | Task success |", "|---|---|---:|---:|---|",
    ]
    for row in rows[:len(CASES)]:
        report.append(f"| {row['panel']} | {row['condition']} | {row['capped_v1']:.3f} | {row['hierarchy_v1']:.3f} | {row['success']} |")
    report += ["", "## What can and cannot be inferred", "",
        "The A-D-B series tests increasingly prominent headline type while holding the CTA fixed. "
        "The original score ties these designs; the new rule distinguishes them by construction. This is evidence of implemented discrimination, "
        "not independent evidence that the ordering is desirable. C demonstrates an alternative optimization route: shrink the CTA rather than enlarge the headline. "
        "E tests saturation: larger-than-target headlines receive no extra credit. F is a counterexample: a large competing message does not change this rule at all. "
        "Some briefs may legitimately favor a prominent CTA over a headline, so a universal headline-led preference would be inappropriate.", "",
        "The limitation can be shown mathematically: if two scenes have identical measured features, every function of only those features must assign them the same score. "
        "That proves the evaluator cannot distinguish those scenes; it does not prove which scene people prefer. "
        "Likewise, logic alone cannot establish that adding an aesthetic feature improves or worsens human judgment.", "",
        "## Sensitivity and next test", "",
        "We evaluate 72 paired scenes (12 deterministic fixture variants × 6 conditions) and nine weight/target combinations "
        "(.05/.15/.30 and 1.25/1.5/2.0). These are development fixtures, not 72 independent human evaluations. "
        "Raw scores are in `paired_results.csv` and `parameter_sensitivity.csv`. "
        "For independent validation, freeze the rule, collect new layouts including CTA-led briefs, randomize pair order/side, "
        "and ask reviewers who cannot see scores to assess prominence, readability and overall preference. "
        "Compare each reward's pairwise agreement with those judgments, report ties/disagreement, and reserve a separate set for weight tuning. "
        "Aesthetic preference would still not establish click-through or conversion improvement.", "",
        "## Reproduce", "", "```sh", ".venv/bin/python aesthetic_comparison.py", "```", "",
        "Both experimental rewards are offline evaluators. The running playground and MCP retain the original terminal reward. "
        "No model training, deployment, or GitHub operations are part of this study.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(rows[:len(CASES)], indent=2))
    print("Report:", (out / "REPORT.md").resolve())


if __name__ == "__main__":
    main()
