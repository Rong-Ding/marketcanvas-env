"""Paired evaluator ablation. No LLM, optimizer, or changes to live episode rewards.

Run: python reward_comparison.py --seeds 12
Re-running replaces generated files in --output. Source hashes identify local code
even before a Git commit exists. No GitHub account or remote is accessed.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from PIL import Image, ImageDraw

from marketcanvas.rendering import font
from marketcanvas.reward_variants import compare_rewards
from marketcanvas.rewards import WEIGHTS
from marketcanvas.scenarios import CASES, apply, load_scenario

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measurement(env, **labels):
    result = compare_rewards(env.state)
    return {**labels, "capped_v1": result["capped_v1"], "uncapped_v1": result["uncapped_v1"],
            "success": result["success"], "essential_constraints_pass": result["essential_constraints_pass"],
            **result["components"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/reward-comparison"))
    args = parser.parse_args()
    if not 1 <= args.seeds <= 1000:
        parser.error("seeds must be 1..1000")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    fixtures = [measurement(load_scenario(case, seed), seed=seed, case=case)
                for seed in range(args.seeds) for case in CASES]
    write_csv(out / "paired_fixtures.csv", fixtures)

    # Independent final scenes: all 256 representable grayscale CTA text colors.
    # Sort by measured contrast for analysis; brightness is not itself contrast.
    sweep = []
    for gray in range(256):
        env = load_scenario()
        code = f"#{gray:02X}{gray:02X}{gray:02X}"
        apply(env, {"op": "update_element", "id": "e3", "properties": {"text_color": code}})
        result = compare_rewards(env.state)
        sweep.append(measurement(env, gray=gray, hex=code,
                                 contrast_ratio=result["elements"]["cta"]["contrast"]))
    write_csv(out / "contrast_sweep.csv", sweep)
    below = sorted((r for r in sweep if r["contrast_ratio"] < env.state.task.min_contrast),
                   key=lambda r: r["contrast_ratio"])
    pairs = list(zip(below, below[1:]))
    improvements = {variant: sum(b[variant] > a[variant] for a, b in pairs)
                    for variant in ("capped_v1", "uncapped_v1")}

    font_rows = []
    for size in range(8, 25):
        env = load_scenario()
        apply(env, {"op": "update_element", "id": "e1", "properties": {"font_size": size}})
        font_rows.append(measurement(env, font_size=size))
    write_csv(out / "font_sweep.csv", font_rows)

    # Unlike the original duplicate fixture, this isolates uniqueness from occlusion.
    env = load_scenario()
    cta = next(e for e in env.state.elements if e.role == "cta")
    spec = {k: v for k, v in cta.model_dump().items() if k not in {"id", "creation_index"}}
    spec.update(x=272, y=520)
    apply(env, {"op": "add_element", "element": spec})
    counterexample = measurement(env, case="nonoverlapping_duplicate_cta")
    env.save_png(out / "duplicate_counterexample.png")
    (out / "duplicate_counterexample.json").write_text(json.dumps(
        {"observation": env.observe(), "comparison": compare_rewards(env.state)}, indent=2) + "\n")

    # One actual repair trajectory, with alternative scores stored separately.
    env = load_scenario("low_contrast")
    progress = [measurement(env, edit=0, hex="#FFFFFF")]
    sheet = Image.new("RGB", (1056, 616), "#EEF2F6")
    for i, gray in enumerate((255, 224, 192, 160, 128, 96)):
        if i:
            code = f"#{gray:02X}{gray:02X}{gray:02X}"
            apply(env, {"op": "update_element", "id": "e3", "properties": {"text_color": code}})
            progress.append(measurement(env, edit=i, hex=code))
        row = progress[-1]
        tile = Image.new("RGB", (352, 308), "#EEF2F6")
        tile.paste(Image.fromarray(env.render()).resize((320, 240)), (16, 8))
        draw = ImageDraw.Draw(tile)
        draw.text((16, 253), f"CTA text {row['hex']}", font=font(15), fill="#172033")
        draw.text((16, 277), f"Capped {row['capped_v1']:.3f}   Uncapped {row['uncapped_v1']:.3f}",
                  font=font(14), fill="#172033")
        sheet.paste(tile, ((i % 3) * 352, (i // 3) * 308))
    sheet.save(out / "contrast_repairs.png")
    apply(env, {"op": "submit"})
    (out / "repair_trajectory.json").write_text(json.dumps(env.export_trajectory(), indent=2) + "\n")
    write_csv(out / "repair_scores.csv", progress)

    essential_failures = [r for r in fixtures if not r["essential_constraints_pass"]]
    positive_failures = {v: sum(r[v] > 0 for r in essential_failures) for v in improvements}
    source_paths = sorted((ROOT / "marketcanvas").glob("*.py")) + [Path(__file__), ROOT / "tests/test_reward_variants.py"]
    metadata = {
        "experiment": "validity_cap_ablation_v1", "variants": list(improvements), "weights": WEIGHTS,
        "task": load_scenario().state.task.model_dump(), "fixture_seeds": list(range(args.seeds)),
        "fixture_count": len(fixtures), "contrast_samples": len(sweep), "below_threshold_samples": len(below),
        "strictly_improving_adjacent_pairs": improvements, "below_threshold_adjacent_pairs": len(pairs),
        "essential_failure_fixtures": len(essential_failures), "positive_scores_on_essential_failures": positive_failures,
        "counterexample": counterexample, "action_mode": "semantic", "observation_mode": "semantic_json",
        "policy": "scripted; no LLM or optimizer", "max_edit_steps": 40, "live_reward": "capped_v1 terminal only",
        "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("pillow", "numpy", "pydantic")},
        "source_sha256": {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = [
        "# Reward comparison: can a score reflect incremental progress?", "",
        "**Finding:** removing the validity cap reveals gradual contrast improvements, but allows high scores for invalid designs. Neither reward measures excellent marketing design.", "",
        "## Method", "",
        "The baseline is `capped_v1`; `uncapped_v1` removes only its essential-validity cap. "
        "Both use the same rendered measurements and weights: 45% constraints, 20% visibility, 20% contrast, 15% layout. "
        "We rescore identical scenes, keeping task success as an independent, unchanged outcome. "
        "The uncapped reward is graded in contrast and alignment, not globally continuous: exact text/color, visibility and spacing still have thresholds.", "",
        f"We evaluate {len(fixtures)} paired development fixtures ({args.seeds} deterministic variants × {len(CASES)} conditions), "
        "all 256 grayscale CTA text colors, 17 headline font sizes, and one nonoverlapping duplicate-CTA counterexample. "
        "A separate six-state repair sequence shows how score evolves through actual edits. No model is run or trained. "
        "These cases were selected with knowledge of the reward and are not independent evidence of human design preference or learning efficiency.", "",
        "## Gradual contrast repair", "",
        "Only CTA text color changes. The headline, yellow button, position and task stay fixed.", "",
        "| CTA text | Capped | Uncapped | Task success |", "|---|---:|---:|---|",
    ]
    for r in progress:
        report.append(f"| {r['hex']} | {r['capped_v1']:.3f} | {r['uncapped_v1']:.3f} | {r['success']} |")
    report += ["", "![Identical designs scored two ways](contrast_repairs.png)", "",
        f"Of {len(pairs)} adjacent increases in measured contrast below 4.5:1, the capped score increases "
        f"{improvements['capped_v1']} times; the uncapped score increases {improvements['uncapped_v1']} times. "
        "Pairs are ordered by measured contrast, not grayscale value. This is a deterministic sensitivity check, not a statistical sample. "
        "Above the threshold, the contrast component saturates intentionally.", "",
        "## Costs and counterexamples", "",
        f"Among {len(essential_failures)} fixtures with essential failures, the capped reward gives "
        f"{positive_failures['capped_v1']} positive scores; the uncapped reward gives {positive_failures['uncapped_v1']}. "
        "Positive is not synonymous with valid; success must be reported separately.", "",
        "A second CTA placed below the original, without covering it, scores "
        f"**{counterexample['uncapped_v1']:.3f} uncapped versus {counterexample['capped_v1']:.3f} capped**, despite violating uniqueness. "
        "Uniqueness is represented by the cap, not a separate weighted component, so removing the cap removes that penalty entirely. "
        "The font sweep also retains a plateau below 16 pixels. Both versions still award full credit to unrelated extra copy. "
        "The ablation therefore supports a narrower conclusion than 'the new reward is better'.", "",
        "## Score, progress, and training are different", "",
        "This experiment is offline rescoring. The playground, MCP and `env.step` continue to use the original capped terminal reward. "
        "`repair_trajectory.json` records zero intermediate rewards and the baseline terminal reward; alternative previews are in `repair_scores.csv`. "
        "To use the alternative in a future trainer, select it at episode reset, keep it fixed through the episode, evaluate it at termination, "
        "and log its version/configuration. That rollout integration is not implemented here.", "",
        "A final reward can depend on all accumulated design improvements without paying reward on every action. "
        "Naively summing positive improvements lets an agent damage and repair the same element repeatedly. "
        "Potential-based shaping instead adds `gamma * Phi(next_state) - Phi(state)` to the original reward, "
        "with zero terminal potential for this episodic formulation and the same discount as the trainer. "
        "This preserves the underlying return ordering under the relevant assumptions; replacing the terminal objective with our uncapped score does not. "
        "Shaping is not implemented. See [Ng et al., 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).", "",
        "## Reproduce", "", "```sh", f".venv/bin/python reward_comparison.py --seeds {args.seeds}",
        ".venv/bin/python replay.py results/reward-comparison/repair_trajectory.json", "```", "",
        "Raw CSV files, a counterexample scene, trajectory, and source/dependency metadata accompany this report. "
        "Use `--output` for a separate run directory. See [the research notes](../../RESEARCH_NOTES.md) for human validation and prospective training design.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({k: metadata[k] for k in ("fixture_count", "below_threshold_samples", "strictly_improving_adjacent_pairs",
                                               "positive_scores_on_essential_failures", "counterexample")}, indent=2))
    print("Report:", (out / "REPORT.md").resolve())


if __name__ == "__main__":
    main()
