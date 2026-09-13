"""Descriptive single-reviewer agreement; never infer human ratings from a model."""
import json
from pathlib import Path


def choice(left, right, tolerance=1e-6):
    return "tie" if abs(left-right) <= tolerance else "left" if left > right else "right"


def analyze(study, record):
    study = Path(study)
    predictions = json.loads((study / "predictions.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    tolerance = protocol["analysis"]["model_tie_tolerance"]
    if not record.get("completed_at") or len(record["ratings"]) != len(predictions):
        raise ValueError("Complete the pilot before revealing predictions")
    rows, summaries = [], []
    variants = {"original": ("primary", "baseline_score"), "viewer": ("primary", "score")}
    variants.update({f"sensitivity:{s}": (s, "score") for s in protocol["sensitivity"] if s != "primary"})
    for pair_id, prediction in predictions.items():
        rating = record["ratings"][pair_id]
        row = {"pair": pair_id, "family": prediction["family"],
               "discoverability": rating["discoverability"], "preference": rating["preference"]}
        for name, (setting, metric) in variants.items():
            def value(side):
                return prediction[side]["primary"][metric] if setting == "primary" else prediction[side]["sensitivity"][setting][metric]
            row[name] = choice(value("left"), value("right"), tolerance)
        row.update(D2_left=prediction["left"]["primary"]["D2"], D2_right=prediction["right"]["primary"]["D2"])
        rows.append(row)
    for question in ("discoverability", "preference"):
        eligible = [r for r in rows if r[question] != "unsure"]
        directional = [r for r in eligible if r[question] in ("left", "right")]
        for model in variants:
            agreement = sum(r[question] == r[model] for r in eligible)
            decisive = [r for r in directional if r[model] != "tie"]
            summaries.append({"question": question, "model": model,
                              "agreements": agreement, "rated_non_unsure": len(eligible),
                              "agreement_fraction": agreement/len(eligible) if eligible else None,
                              "human_ties": len(eligible)-len(directional), "unsure": len(rows)-len(eligible),
                              "model_ties": sum(r[model] == "tie" for r in eligible),
                              "decisive_on_human_directional": len(decisive), "human_directional": len(directional),
                              "directional_agreements": sum(r[question] == r[model] for r in decisive)})
    return {"study_id": record["study_id"], "freeze_hash": record["freeze_hash"], "reviewer_count": 1,
            "pair_count": len(rows), "summaries": summaries, "pairs": rows,
            "interpretation": "Single reviewer involved in study design; descriptive development evidence only. No gaze, response-time, comprehension, or marketing-outcome measurements."}


def write_report(study, record, destination):
    result = analyze(study, record)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Viewer pilot: completed single-reviewer comparison", "", result["interpretation"], "",
             f"Frozen study: `{result['freeze_hash']}`. Human ratings are stored separately; free-text notes are omitted here.", "",
             "| Judgment | Evaluator | Exact agreements | Human ties | Unsure | Model ties |", "|---|---|---:|---:|---:|---:|"]
    for r in result["summaries"]:
        lines.append(f"| {r['question']} | {r['model']} | {r['agreements']}/{r['rated_non_unsure']} | {r['human_ties']} | {r['unsure']} | {r['model_ties']} |")
    lines += ["", "A model tie agrees only with a reviewer tie. Unsure responses are excluded from the denominator and reported. "
              "The original evaluator ties all these valid designs; comparison therefore tests added discrimination. "
              "Agreement does not establish causal effects or preference in other people. Do not treat 12 related pairs as 12 independent participants.", "",
              "## Pair-level findings", "", "| Pair | Manipulation | Discoverability rating | Preference rating | Original prediction | Viewer prediction |", "|---|---|---|---|---|---|"]
    for r in result["pairs"]:
        lines.append(f"| {r['pair']} | {r['family']} | {r['discoverability']} | {r['preference']} | {r['original']} | {r['viewer']} |")
    lines += ["", "## Interpretation", "", "Inspect where discoverability and preference diverge, and whether model rankings depend on parameter settings. "
              "The task-aware reviewer searches for a named CTA; the model has no semantics and predicts early visual exploration. "
              "Disagreement may therefore reflect model assumptions, different constructs, or both. No parameters were fitted to these ratings. "
              "Any revised model needs a separately labeled exploratory analysis and new evaluation material.", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))
    return result
