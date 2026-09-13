"""Scripted fixtures use the same public actions as an agent. No trained policy required."""
from .env import MarketCanvasEnv
from .models import TaskSpec


def apply(env, action):
    result = env.step(action)
    if result[4]["error"]:
        raise ValueError(result[4]["error"])
    return result


def build_reference(seed=0, task=None):
    if task is None:
        headlines = ["Summer Sale", "Weekend Offers", "Fresh Arrivals", "Autumn Collection"]
        task = TaskSpec(headline=headlines[seed % 4], cta=["Shop now", "Explore offers", "Discover more"][seed % 3])
    env = MarketCanvasEnv(task=task)
    env.reset(seed=seed)
    y = 72 + (seed % 5)*8
    cta_width = 256 + (seed % 3)*16
    left = 100 if task.alignment == "left" else (800-cta_width)//2
    specs = [
        dict(type="text", role="headline", content=task.headline, x=100, y=y, width=600, height=90, font_size=42),
        dict(type="image", color="#DBEAFE", content="Product image placeholder", x=100, y=230, width=600, height=150),
        dict(type="shape", role="cta", color=task.cta_color, content=task.cta, x=left, y=440, width=cta_width, height=72, font_size=28),
    ]
    for spec in specs:
        apply(env, {"op": "add_element", "element": spec})
    return env


def perturb(env, case):
    h = next(e for e in env.state.elements if e.role == "headline")
    c = next(e for e in env.state.elements if e.role == "cta")
    cases = {
        "low_contrast": {"op": "update_element", "id": c.id, "properties": {"text_color": "#FFFFFF"}},
        "wrong_headline": {"op": "update_element", "id": h.id, "properties": {"content": "Wrong campaign"}},
        "tiny_headline": {"op": "update_element", "id": h.id, "properties": {"font_size": 8}},
        "clipped_headline": {"op": "update_element", "id": h.id, "properties": {"x": 390, "width": 20}},
        "missing_cta": {"op": "delete_element", "id": c.id},
        "misaligned_cta": {"op": "move_element", "id": c.id, "x": c.x+120, "y": c.y},
        "covered_headline": {"op": "add_element", "element": {"type": "image", "color": "#FFFFFF", "x": h.x, "y": h.y, "width": h.width, "height": h.height, "z_index": 20}},
        "duplicate_cta": {"op": "add_element", "element": {k: v for k,v in c.model_dump().items() if k not in ("id", "creation_index")}},
        "neutral_image_color": {"op": "update_element", "id": "e2", "properties": {"color": "#D1FAE5"}},
        "irrelevant_decoration": {"op": "add_element", "element": {"type": "text", "content": "Unrelated marketing message", "x": 100, "y": 175, "width": 600, "height": 40, "font_size": 22}},
    }
    if case == "reference":
        return env
    if case not in cases:
        raise ValueError(f"Unknown scenario: {case}")
    apply(env, cases[case])
    return env


CASES = ["reference", "low_contrast", "wrong_headline", "tiny_headline", "clipped_headline", "missing_cta",
         "misaligned_cta", "covered_headline", "duplicate_cta", "neutral_image_color", "irrelevant_decoration"]


def load_scenario(case="reference", seed=0, task=None):
    """Prepare a scene outside the participant's 40-action budget."""
    if case not in CASES:
        raise ValueError(f"Unknown scenario: {case}")
    env = perturb(build_reference(seed, task), case)
    env.begin_prepared_episode()
    return env
