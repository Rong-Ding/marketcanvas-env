"""Deterministic evaluator studies, not LLM performance or training experiments.

Run: python experiments.py --seeds 12 --output results/experiments
All scenarios use the public action API. Expectations are directional/invariance checks.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path
from PIL import Image, ImageDraw
from marketcanvas.scenarios import build_reference, perturb, apply, CASES
from marketcanvas.rewards import evaluate
from marketcanvas.rendering import font


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(env, n=30):
    def measure(fn):
        fn()
        times = []
        for _ in range(n):
            start = time.perf_counter()
            fn()
            times.append(1000*(time.perf_counter()-start))
        return {"median_ms": statistics.median(times), "p95_ms": sorted(times)[int(.95*(n-1))], "samples": n}
    return {"observe": measure(env.observe), "json_serialization": measure(lambda: json.dumps(env.observe())),
            "render": measure(env.render), "evaluate_uncached": measure(lambda: evaluate(env.state))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--output", default="results/experiments")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 1000:
        parser.error("seeds must be 1..1000")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, thumbnails = [], []
    defects = set(CASES) - {"reference", "neutral_image_color", "irrelevant_decoration"}
    for seed in range(args.seeds):
        reference_score = build_reference(seed).current_reward()["score"]
        for case in CASES:
            env = perturb(build_reference(seed), case)
            result = env.current_reward()
            expected = "decrease" if case in defects else "equal" if case in {"reference", "neutral_image_color"} else "limitation_probe"
            passed = result["score"] < reference_score if expected == "decrease" else result["score"] == reference_score if expected == "equal" else None
            rows.append({"seed": seed, "case": case, "score": result["score"], "presence_baseline": result["presence_baseline"],
                         "delta": round(result["score"]-reference_score,6), "expectation": expected, "expectation_met": passed,
                         "success": result["success"], **result["components"]})
            if seed == 0:
                env.save_png(out / f"{case}.png")
                (out / f"{case}.json").write_text(json.dumps({"state": env.observe(), "evaluation": result}, indent=2)+"\n")
                thumb = Image.fromarray(env.render()).resize((320,240))
                tile = Image.new("RGB", (340,292), "#EEF2F6")
                tile.paste(thumb,(10,10))
                d = ImageDraw.Draw(tile)
                d.text((10,255),case.replace("_"," "),font=font(14),fill="#172033")
                d.text((10,274),f"reward {result['score']:.3f} | baseline {result['presence_baseline']:.3f}",font=font(12),fill="#536277")
                thumbnails.append(tile)
    write_csv(out / "paired_results.csv", rows)
    sheet = Image.new("RGB", (340*3,292*4), "#EEF2F6")
    for i, tile in enumerate(thumbnails):
        sheet.paste(tile,((i%3)*340,(i//3)*292))
    sheet.save(out / "comparison.png")
    sweep=[]
    for gray in list(range(0,256,8))+[255]:
        env = build_reference()
        apply(env,{"op":"update_element","id":"e3","properties":{"text_color":f"#{gray:02X}{gray:02X}{gray:02X}"}})
        r=env.current_reward()
        sweep.append({"gray":gray,"contrast_ratio":r["elements"]["cta"]["contrast"],"contrast_component":r["components"]["contrast"],"score":r["score"],"success":r["success"]})
    write_csv(out / "contrast_sweep.csv",sweep)
    alignment=[]
    for dx in range(0,241,20):
        env=build_reference()
        apply(env,{"op":"move_element","id":"e3","x":272+dx,"y":440})
        r=env.current_reward()
        alignment.append({"horizontal_offset":dx,"layout_component":r["components"]["layout"],"score":r["score"]})
    write_csv(out / "alignment_sweep.csv",alignment)
    timings=benchmark(build_reference())
    contract=[r for r in rows if r["expectation"]!="limitation_probe"]
    detected={case: all(r["presence_baseline"]<1 for r in rows if r["case"]==case) for case in sorted(defects)}
    metadata={"seeds":args.seeds,"case_count":len(CASES),"contract_checks":len(contract),
              "passed":sum(r["expectation_met"] for r in contract),"weak_baseline_detected_defect_families":sum(detected.values()),
              "defect_families":len(defects),"timings":timings,"python":platform.python_version(),"platform":platform.system(),
              "architecture":platform.machine(),"dependencies":{p:importlib.metadata.version(p) for p in ["pillow","numpy","pydantic","mcp","gymnasium"]},
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("marketcanvas").glob("*.py"))}}
    (out / "metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    report=["# Evaluator mini-experiments", "", "These are controlled tests of a reward proxy. No LLM was run or trained.", "",
            f"## Design\n\n{args.seeds} deterministic fixture variants × {len(CASES)} conditions = {len(rows)} evaluated canvases. "
            "Variants change headline, CTA wording, heading position, and button width. Each intervention is paired with its own reference. "
            "Seeds label fixture variants; transitions contain no randomness. The cases were designed alongside the evaluator and are development checks, not an independent test set. "
            "No p-values or population-level confidence intervals are claimed.", "",
            "## Results", "", "| Condition | Mean reward | Presence-only baseline | Expectation |", "|---|---:|---:|---|"]
    for case in CASES:
        group=[r for r in rows if r["case"]==case]
        report.append(f"| {case} | {statistics.mean(r['score'] for r in group):.3f} | {statistics.mean(r['presence_baseline'] for r in group):.3f} | {group[0]['expectation']} |")
    report += ["",f"Directional/invariance checks: **{metadata['passed']}/{len(contract)}**. "
               f"The presence-only baseline reduces reward for **{sum(detected.values())}/{len(defects)}** defect families; "
               "its simplicity is intentional and it is not a competitive state-of-the-art baseline.", "",
               "![Controlled canvas variants](comparison.png)", "", "## Interpretation and unresolved failures", "",
               "The structured evaluator catches explicit content errors, poor contrast, clipping, small text, occlusion, duplicates, and misalignment in these fixtures. "
               "The neutral image recoloring leaves reward unchanged. However, unrelated extra text still receives full reward. "
               "The reward validates required content and basic layout, not persuasion, overall composition, image relevance, or every decorative text element. "
               "Counting these checks as model generalization results would be incorrect.", "",
               "The contrast sweep reveals a deliberate discontinuity: essential-constraint failures cap quality at 0.49 (reward -0.02), "
               "so the final reward can jump when contrast crosses the threshold. Component scores retain more detail but are diagnostic only. "
               "This protects validity at the cost of a less informative terminal learning signal. The alignment sweep is graded. "
               "Neither sweep demonstrates learning efficiency.", "", "## Local microbenchmark", "",
               "One process, three elements, 800×600 RGB; 30 timed repetitions after one warm-up per operation. "
               "These are local timings, not a distributed throughput forecast. Step timings would also include optional trajectory copying and terminal evaluation.", "",
               "| Operation | Median ms | p95 ms |", "|---|---:|---:|"]
    for name,t in timings.items():
        report.append(f"| {name} | {t['median_ms']:.3f} | {t['p95_ms']:.3f} |")
    report += ["", "## Next experiment", "",
               "Ask independent reviewers to compare new, held-out canvas pairs without seeing the reward. Separate task correctness, readability, and aesthetic preference. "
               "Measure rubric agreement and inspect disagreements with the heuristic before tuning weights. Then test a fixed LLM on repair tasks under a fixed action budget; "
               "compare terminal-only feedback with diagnostic feedback while keeping model, prompts, and task variants fixed. "
               "Agent behavior and reward validity are distinct studies.", "",
               "## Reproduce", "", "```sh", f"python experiments.py --seeds {args.seeds}", "```", "",
               "See `paired_results.csv`, `contrast_sweep.csv`, `alignment_sweep.csv`, and `metadata.json` for raw results and environment versions."]
    (out / "REPORT.md").write_text("\n".join(report)+"\n")
    print(json.dumps(metadata,indent=2))
    print("Report:", (out / "REPORT.md").resolve())
    if metadata["passed"] != len(contract):
        raise SystemExit("Some expected relationships failed; inspect the raw results")


if __name__=="__main__":
    main()
