import copy

import pytest

from marketcanvas import MarketCanvasEnv
from marketcanvas.reward_variants import compare_rewards, evaluate_uncapped, evaluate_hierarchy
from marketcanvas.scenarios import CASES, apply, load_scenario


def test_contrast_repairs_receive_credit_below_threshold():
    rows = []
    for gray in (255, 224, 192, 160):
        env = load_scenario()
        apply(env, {"op": "update_element", "id": "e3",
                    "properties": {"text_color": f"#{gray:02X}{gray:02X}{gray:02X}"}})
        rows.append(compare_rewards(env.state))
    assert all(not r["essential_constraints_pass"] for r in rows)
    assert {r["capped_v1"] for r in rows} == {-.02}
    assert all(b["uncapped_v1"] > a["uncapped_v1"] for a, b in zip(rows, rows[1:]))


@pytest.mark.parametrize("case", CASES)
def test_rescoring_does_not_change_episode_or_success(case):
    env = load_scenario(case)
    before, trace = env.state_hash(), copy.deepcopy(env.export_trajectory())
    baseline = env.current_reward()
    alternative = evaluate_uncapped(env.state)
    comparison = compare_rewards(env.state)
    assert alternative["success"] == baseline["success"]
    assert alternative["essential_constraints_pass"] == baseline["essential_constraints_pass"]
    assert -1 <= alternative["score"] <= 1
    assert alternative["score"] == comparison["uncapped_v1"]
    assert comparison["capped_v1"] == baseline["score"]
    assert env.state_hash() == before and env.export_trajectory() == trace
    assert env.step({"op": "submit"})[1] == baseline["score"]


def test_counterexamples_document_tradeoff_not_universal_improvement():
    # Matching labels/color/geometry can compensate for a wrong campaign in the ablation.
    result = compare_rewards(load_scenario("wrong_headline").state)
    assert result["capped_v1"] < 0 < result["uncapped_v1"]
    assert not result["success"]
    # Isolate uniqueness from occlusion: a second CTA outside the first one's box.
    env = load_scenario()
    cta = next(e for e in env.state.elements if e.role == "cta")
    spec = {k: v for k, v in cta.model_dump().items() if k not in {"id", "creation_index"}}
    spec.update(x=272, y=520)
    apply(env, {"op": "add_element", "element": spec})
    result = compare_rewards(env.state)
    assert result["uncapped_v1"] == 1 and not result["essential_constraints_pass"]
    assert result["capped_v1"] < 0
    assert evaluate_uncapped(MarketCanvasEnv().state)["score"] == -1


def test_smoother_does_not_mean_every_requirement_is_continuous():
    scores = []
    for size in (12, 13, 14, 15, 16):
        env = load_scenario()
        apply(env, {"op": "update_element", "id": "e1", "properties": {"font_size": size}})
        scores.append(evaluate_uncapped(env.state)["score"])
    assert len(set(scores[:-1])) == 1
    assert scores[-1] > scores[-2]


def test_hierarchy_discriminates_valid_sizes_but_has_alternative_routes():
    scores = []
    for size in (28, 35, 42):
        env = load_scenario()
        apply(env, {"op": "update_element", "id": "e1", "properties": {"font_size": size}})
        assert env.current_reward()["score"] == 1
        scores.append(evaluate_hierarchy(env.state)["score"])
    assert scores[0] < scores[1] < scores[2]
    # The same numerical hierarchy can result from reducing CTA readability margin.
    apply(env, {"op": "update_element", "id": "e1", "properties": {"font_size": 28}})
    apply(env, {"op": "update_element", "id": "e3", "properties": {"font_size": 16}})
    assert evaluate_hierarchy(env.state)["score"] == 1
    for case in ("low_contrast", "wrong_headline", "duplicate_cta"):
        assert evaluate_hierarchy(load_scenario(case).state)["score"] < 0
    assert evaluate_hierarchy(load_scenario("irrelevant_decoration").state)["score"] == 1


@pytest.mark.parametrize("ratio", [1.25, 1.5, 2.0])
def test_hierarchy_zero_weight_recovers_baseline(ratio):
    for case in CASES:
        env = load_scenario(case)
        assert evaluate_hierarchy(env.state, style_weight=0, target_ratio=ratio)["score"] == env.current_reward()["score"]
