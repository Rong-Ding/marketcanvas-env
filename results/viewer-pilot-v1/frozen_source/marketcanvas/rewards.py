"""Transparent task-specific proxy, intentionally not a learned aesthetic judgment."""
from .models import State
from .rendering import render_scene

WEIGHTS = {"constraints": .45, "visibility": .20, "contrast": .20, "layout": .15}


def weighted_quality(components):
    return sum(weight * components[name] for name, weight in WEIGHTS.items())


def norm(text):
    return " ".join(text.casefold().split())


def presence_baseline(state: State):
    """Deliberately weak ablation: roles and CTA fill, ignoring actual readable content."""
    h = any(e.role == "headline" for e in state.elements)
    c = any(e.role == "cta" for e in state.elements)
    yellow = any(e.role == "cta" and (e.color or "").upper() == state.task.cta_color.upper() for e in state.elements)
    return 2 * (h+c+yellow)/3 - 1


def evaluate(state: State):
    _, metrics = render_scene(state)
    hs = [e for e in state.elements if e.role == "headline" and e.type == "text"]
    cs = [e for e in state.elements if e.role == "cta" and e.type == "shape"]
    # Each required slot counts once. Duplicate roles violate the simple one-headline/one-CTA task.
    h, c = (hs[0] if hs else None), (cs[0] if cs else None)
    content = [bool(h and norm(h.content) == norm(state.task.headline)),
               bool(c and norm(c.content) == norm(state.task.cta)),
               bool(c and c.color.upper() == state.task.cta_color.upper())]
    details = {}
    visible, accessible = [], []
    issues = []
    for role, element in (("headline", h), ("cta", c)):
        m = metrics.get(element.id, {}) if element else {}
        readable_geometry = bool(m.get("fits") and m.get("visible_fraction", 0) >= .99 and m.get("font_size", 0) >= 16)
        ratio = m.get("contrast", 0)
        visible.append(float(readable_geometry))
        accessible.append(min(1.0, max(0.0, (ratio-1)/(state.task.min_contrast-1))) if state.task.min_contrast > 1 else float(bool(m)))
        details[role] = {"element_id": element.id if element else None, **m, "readable_geometry": readable_geometry,
                         "contrast_pass": ratio >= state.task.min_contrast}
        if not readable_geometry:
            issues.append(f"{role}: missing, clipped, obscured, or below 16px")
        if ratio < state.task.min_contrast:
            issues.append(f"{role}: contrast {ratio:.2f}:1 below {state.task.min_contrast}:1")
    alignment, order = 0.0, 0.0
    if h and c:
        if state.task.alignment == "center":
            alignment = max(0, 1-(abs(h.x+h.width/2-400)+abs(c.x+c.width/2-400))/400)
        else:
            alignment = max(0, 1-abs(h.x-c.x)/400)
        order = float(c.y >= h.y+h.height+16)
    unique = sum(e.role == "headline" for e in state.elements) == 1 and sum(e.role == "cta" for e in state.elements) == 1
    components = {"constraints": sum(content)/3, "visibility": sum(visible)/2,
                  "contrast": sum(accessible)/2, "layout": (alignment+order)/2}
    quality = weighted_quality(components)
    essential = all(content) and all(visible) and all(v["contrast_pass"] for v in details.values()) and unique
    if not all(content):
        issues.append("Required headline, CTA label, or CTA fill does not match the task")
    if not unique:
        issues.append("Exactly one headline role and one CTA role are required")
    if not essential:
        quality = min(quality, .49)
    if components["layout"] < .999:
        issues.append("Headline and CTA do not meet the requested alignment and spacing")
    return {"score": round(2*quality-1, 6), "components": components, "elements": details,
            "success": essential and components["layout"] >= .999,
            "essential_constraints_pass": essential, "issues": issues,
            "presence_baseline": round(presence_baseline(state), 6)}
