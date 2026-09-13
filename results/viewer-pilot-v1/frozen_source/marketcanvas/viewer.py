"""Exploratory, role-blind region viewer. No learned gaze model or human calibration.

Region segmentation is privileged simulator information. Appearance is measured
from pixels; region selection never receives task text or semantic roles. Exact
finite-state propagation marginalizes revisit histories for 1--3 model glances.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .models import State, ordered
from .rendering import luminance, render_scene
from .rewards import evaluate, weighted_quality

VERSION = "viewer_v1"


@dataclass(frozen=True)
class ViewerConfig:
    size_weight: float = .5
    beta: float = 2.0
    distance_weight: float = 1.0
    revisit_penalty: float = 1.0
    start_x: float = .5
    start_y: float = .5

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(x) for x in values.values()):
            raise ValueError("Viewer settings must be finite")
        if not 0 <= self.size_weight <= 1 or not 0 <= self.start_x <= 1 or not 0 <= self.start_y <= 1:
            raise ValueError("Weights/start coordinates must be in [0,1]")
        if any(not 0 <= x <= 10 for x in (self.beta, self.distance_weight, self.revisit_penalty)):
            raise ValueError("Viewer strengths must be in [0,10]")


def extract_regions(state: State):
    """Visible support = pixels changed if one element is removed from the scene.

    Area is the number of affected rendered pixels (ink for transparent text),
    not its editable box. Contrast is mean nonzero local luminance edge strength
    on that support. Occluded/invisible elements have no support. The removal
    test is a rendering attribution heuristic, not a human segmentation model.
    """
    frame = np.asarray(render_scene(state, measure=False)[0])
    lum = luminance(frame)
    edges = np.zeros(lum.shape)
    dx, dy = np.abs(np.diff(lum, axis=1)), np.abs(np.diff(lum, axis=0))
    edges[:, :-1] = np.maximum(edges[:, :-1], dx)
    edges[:, 1:] = np.maximum(edges[:, 1:], dx)
    edges[:-1, :] = np.maximum(edges[:-1, :], dy)
    edges[1:, :] = np.maximum(edges[1:, :], dy)
    regions = []
    for e in ordered(state):
        other = state.model_copy(update={"elements": tuple(x for x in state.elements if x.id != e.id)})
        removed = np.asarray(render_scene(other, measure=False)[0])
        support = np.any(frame != removed, axis=2)
        ys, xs = np.where(support)
        if not len(xs):
            continue
        local_edges = edges[support]
        local_edges = local_edges[local_edges > 0]
        regions.append({"id": e.id, "area": int(len(xs)),
                        "contrast": float(local_edges.mean()) if len(local_edges) else 0.0,
                        "x": float(xs.mean()/state.width), "y": float(ys.mean()/state.height)})
    max_size = max((math.sqrt(r["area"]) for r in regions), default=1)
    max_contrast = max((r["contrast"] for r in regions), default=0) or 1
    for r in regions:
        r["size_normalized"] = math.sqrt(r["area"])/max_size
        r["contrast_normalized"] = r["contrast"]/max_contrast
    return regions


def predict_visits(regions, config=ViewerConfig(), glances=3):
    """Return per-region first-visit cumulative probabilities; no target input.

    States are (current region, visited bitmask). All regions, including already
    visited ones, remain selectable. Independent of element names, roles or task.
    """
    if type(glances) is not int or not 1 <= glances <= 3:
        raise ValueError("This bounded pilot supports 1..3 glances")
    n = len(regions)
    if n > 24:
        raise ValueError("At most 24 regions")
    visits = {r["id"]: [] for r in regions}
    if not n:
        return {"visit_probability": visits, "first_glance": {}, "mass_by_glance": [1.0]*glances}
    prominence = np.array([config.size_weight*r["size_normalized"] +
                          (1-config.size_weight)*r["contrast_normalized"] for r in regions])
    positions = np.array([(r["x"], r["y"]) for r in regions])
    # Distance uses physical aspect ratio and the canvas diagonal (800 x 600).
    def distance(gaze):
        return np.linalg.norm((positions-gaze)*np.array([.8, .6]), axis=1)
    base = {i: config.beta*prominence-config.distance_weight*distance(p) for i, p in enumerate(positions)}
    base[-1] = config.beta*prominence-config.distance_weight*distance((config.start_x, config.start_y))
    distribution = {(-1, 0): 1.0}
    masses, first = [], {}
    for t in range(glances):
        following = {}
        for (current, seen), mass in distribution.items():
            logits = base[current]-config.revisit_penalty*np.array([bool(seen & (1 << i)) for i in range(n)])
            probabilities = np.exp(logits-logits.max())
            probabilities /= probabilities.sum()
            for i, probability in enumerate(probabilities):
                key = (i, seen | (1 << i))
                following[key] = following.get(key, 0.0) + mass*float(probability)
        distribution = following
        masses.append(sum(distribution.values()))
        for i, r in enumerate(regions):
            visits[r["id"]].append(sum(m for (_, seen), m in distribution.items() if seen & (1 << i)))
        if t == 0:
            first = {r["id"]: visits[r["id"]][0] for r in regions}
    return {"visit_probability": visits, "first_glance": first, "mass_by_glance": masses}


def evaluate_viewer(state: State, config=ViewerConfig(), alpha=.15, *, regions=None):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0,1]")
    regions = extract_regions(state) if regions is None else regions
    predictions = predict_visits(regions, config)
    ctas = [e for e in state.elements if e.type == "shape" and e.role == "cta"]
    ds = predictions["visit_probability"].get(ctas[0].id, [0.0]*3) if len(ctas) == 1 else [0.0]*3
    baseline = evaluate(state)
    quality = (1-alpha)*weighted_quality(baseline["components"])+alpha*ds[1]
    if not baseline["essential_constraints_pass"]:
        quality = min(quality, .49)
    return {"version": VERSION, "config": asdict(config), "alpha": alpha,
            "score": round(2*quality-1, 6), "baseline_score": baseline["score"],
            "success": baseline["success"], "essential_constraints_pass": baseline["essential_constraints_pass"],
            "D1": ds[0], "D2": ds[1], "D3": ds[2], "regions": regions, **predictions}
