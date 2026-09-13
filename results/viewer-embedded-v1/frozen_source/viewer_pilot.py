"""Build an immutable, local single-reviewer pilot. Never overwrite a frozen study.

python viewer_pilot.py --output results/viewer-pilot-v1
python rating_server.py --study results/viewer-pilot-v1
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random

from marketcanvas import MarketCanvasEnv, TaskSpec
from marketcanvas.models import observation
from marketcanvas.scenarios import apply
from marketcanvas.viewer import ViewerConfig, evaluate_viewer, extract_regions

ROOT = Path(__file__).resolve().parent
FAMILIES = ("image_contrast", "competing_text_size", "cta_width", "image_position")
DEFAULT_OUTPUT = ROOT / "results/viewer-pilot-v1"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_scene(family, seed, alternative):
    task = TaskSpec(headline=("Summer Sale", "Weekend Offers", "Fresh Arrivals")[seed],
                    cta=("Shop now", "View offers", "Explore")[seed])
    env = MarketCanvasEnv(task=task)
    env.reset(seed=seed)
    width = (320 if alternative else 200) if family == "cta_width" else 256
    image = dict(type="image", color="#DBEAFE", content="Product placeholder", x=100, y=230, width=600, height=140)
    if family == "image_contrast":
        image["color"] = "#172033" if alternative else "#DBEAFE"
    if family == "image_position":
        image.update(x=300 if alternative else 60, y=230, width=200, height=140)
    specs = [dict(type="text", role="headline", content=task.headline, x=100, y=55+seed*10,
                  width=600, height=90, font_size=42), image,
             dict(type="shape", role="cta", content=task.cta, color="#FFFF00", x=(800-width)//2,
                  y=420+seed*20, width=width, height=72, font_size=28)]
    if family == "competing_text_size":
        specs.append(dict(type="text", content="Limited time", x=100, y=160, width=600, height=60,
                          font_size=40 if alternative else 18))
    for spec in specs:
        apply(env, {"op": "add_element", "element": spec})
    env.begin_prepared_episode()
    if not env.current_reward()["success"]:
        raise ValueError(f"Invalid planned stimulus: {family}/{seed}/{alternative}")
    return env


def settings():
    default = ViewerConfig()
    return {"primary": default,
            "weaker_prominence": replace(default, beta=1), "stronger_prominence": replace(default, beta=4),
            "no_distance": replace(default, distance_weight=0), "stronger_distance": replace(default, distance_weight=2),
            "no_memory": replace(default, revisit_penalty=0), "stronger_memory": replace(default, revisit_penalty=2),
            "upper_start": replace(default, start_y=.25)}


def verify_study(out):
    protocol = json.loads((out / "protocol.json").read_text())
    if (out / "freeze.sha256").read_text().strip() != canonical_hash(protocol):
        raise ValueError("Study protocol fingerprint changed")
    for name, digest in protocol["files_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"Frozen file changed: {name}")
    return protocol


def build_study(out=DEFAULT_OUTPUT):
    out = Path(out)
    if out.exists():
        return verify_study(out)
    out.mkdir(parents=True)
    (out / "stimuli").mkdir()
    (out / "states").mkdir()
    # Predetermined cases and presentation assignment, never selected by model scores.
    rng = random.Random(20260913)
    conditions = [(family, seed) for family in FAMILIES for seed in range(3)]
    rng.shuffle(conditions)
    swaps = [False]*6+[True]*6
    rng.shuffle(swaps)
    public, predictions = [], {}
    for i, ((family, seed), swapped) in enumerate(zip(conditions, swaps), 1):
        pair_id = f"P{i:02}"
        pair = {"id": pair_id}
        private = {"family": family, "fixture_seed": seed, "swapped": swapped}
        for side, alternative in (("left", swapped), ("right", not swapped)):
            env = make_scene(family, seed, alternative)
            image_name = f"stimuli/{pair_id}_{side}.png"
            env.save_png(out / image_name)
            dump(out / f"states/{pair_id}_{side}.json", env.export_trajectory())
            regions = extract_regions(env.state)
            evaluation = evaluate_viewer(env.state, regions=regions)
            sensitivity = {}
            for label, config in settings().items():
                score = evaluate_viewer(env.state, config, regions=regions)
                sensitivity[label] = {k: score[k] for k in ("score", "D1", "D2", "D3")}
            private[side] = {"alternative": alternative, "primary": evaluation, "sensitivity": sensitivity,
                             "weight_sensitivity": {str(a): evaluate_viewer(env.state, alpha=a, regions=regions)["score"]
                                                    for a in (.05, .15, .30)}}
            pair[side] = image_name
            pair["headline"], pair["cta"] = env.state.task.headline, env.state.task.cta
        public.append(pair)
        predictions[pair_id] = private
    manifest = {"study_id": "viewer-pilot-v1", "pairs": public,
                "reviewer_count": 1, "questions": ["discoverability", "preference"],
                "answers": ["left", "right", "tie", "unsure"]}
    dump(out / "public_manifest.json", manifest)
    dump(out / "predictions.json", predictions)
    # Bundle the exact analysis/UI code and model source so a later edit cannot
    # silently change what was frozen. Participant responses live elsewhere.
    sources = list((ROOT / "marketcanvas").glob("*.py")) + [ROOT / n for n in
              ("viewer_pilot.py", "rating_server.py", "viewer_analysis.py", "marketcanvas/web/rating.html",
               "docs/VIEWER_METHOD.md", "docs/VIEWER_PILOT_WRITEUP.md", "tests/test_viewer_pilot.py")]
    (out / "frozen_source").mkdir()
    for source in sources:
        target = out / "frozen_source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    protocol = {
        "study_id": "viewer-pilot-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": "single-reviewer development pilot; author knows the hypothesis; scores hidden until completion",
        "pair_count": 12, "families": list(FAMILIES), "fixture_seeds": [0, 1, 2],
        "assignment_seed": 20260913, "balanced_side_swaps": 6,
        "model_version": "viewer_v1", "primary": asdict(ViewerConfig()), "primary_glances": 2,
        "alpha": .15, "sensitivity": {k: asdict(v) for k, v in settings().items()},
        "weight_sensitivity": [.05, .15, .30],
        "analysis": {"primary_outcome": "reviewer-perceived CTA discoverability",
                     "secondary_outcome": "overall design preference, analyzed separately",
                     "metric": "strict pairwise agreement, with ties/unsure and coverage reported explicitly",
                     "model_tie_tolerance": 1e-6, "unsure": "excluded from agreement denominator; count reported",
                     "human_tie": "included; agrees only with model tie", "inference": "descriptive only; no population significance claims"},
        "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("pillow", "numpy", "pydantic")},
        "font_sha256": sha(ROOT / "marketcanvas/assets/DejaVuSans.ttf"),
        "files_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    dump(out / "protocol.json", protocol)
    (out / "freeze.sha256").write_text(canonical_hash(protocol)+"\n")
    return verify_study(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = build_study(args.output)
    print(f"Frozen study: {args.output.resolve()} ({protocol['pair_count']} pairs)")
