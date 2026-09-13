import json
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from marketcanvas import MarketCanvasEnv, TaskSpec
from marketcanvas.env import EpisodeFinished
from marketcanvas.gym_adapter import GymCanvasEnv
from marketcanvas.models import State, observation
from marketcanvas.rewards import evaluate
from marketcanvas.rendering import contrast_ratio
from marketcanvas.scenarios import build_reference, perturb, apply


def test_replay_matches_every_transition_and_pixels():
    first=build_reference(seed=7)
    perturb(first,"low_contrast")
    apply(first,{"op":"update_element","id":"e3","properties":{"text_color":"#000000"}})
    apply(first,{"op":"submit"})
    replay=MarketCanvasEnv(task=first.state.task)
    replay.reset(seed=7)
    for t in first.trajectory:
        assert replay.observe()==t["observation"]
        obs,reward,term,trunc,info=replay.step(t["action"])
        assert (obs,reward,term,trunc,info)==(t["next_observation"],t["reward"],t["terminated"],t["truncated"],t["info"])
    assert np.array_equal(first.render(),replay.render())
    assert first.state_hash()==replay.state_hash()


@pytest.mark.parametrize("action",[
    {"op":"move_element","id":"e1","x":799,"y":0},
    {"op":"move_element","id":"e1","x":True,"y":0},
    {"op":"update_element","id":"e1","properties":{"id":"pretend"}},
    {"op":"update_element","id":"e1","properties":{"text_color":"not a color"}},
    {"op":"update_element","id":"e1","properties":{"content":"two\nlines"}},
    {"op":"delete_element","id":"missing"},
    {"op":"add_element","element":{"type":"shape"}},
    {"op":"teleport"}, None,
])
def test_invalid_action_is_atomic_but_costs_one_step(action):
    env=build_reference()
    before=env.state
    result=env.step(action)
    assert result[4]["error"]
    assert env.state.elements==before.elements
    assert env.state.next_id==before.next_id
    assert env.state.steps==before.steps+1
    assert result[1]==0


def test_reward_only_paid_once_and_inspection_is_readonly():
    env=build_reference()
    state_hash=env.state_hash()
    for _ in range(3):
        env.current_reward()["issues"].append("external modification")
        env.observe()["elements"].clear()
    assert env.state_hash()==state_hash
    assert all(t["reward"]==0 for t in env.trajectory)
    assert env.step({"op":"submit"})[1:4]==(1.0,True,False)
    with pytest.raises(EpisodeFinished):
        env.step({"op":"submit"})


def test_intrinsic_budget_terminates_even_for_invalid_actions():
    env=MarketCanvasEnv(max_steps=2)
    assert env.step({"op":"unknown"})[1:4]==(0,False,False)
    assert env.step({"op":"unknown"})[1:4]==(-1,True,False)
    env.reset()
    assert env.state.steps==0 and not env.state.terminated


@pytest.mark.parametrize("case",["low_contrast","wrong_headline","tiny_headline","clipped_headline","missing_cta","covered_headline","duplicate_cta"])
def test_adversarial_required_content_cannot_earn_positive_reward(case):
    r=perturb(build_reference(),case).current_reward()
    assert r["score"]<0 and not r["success"]


def test_invariances_and_known_limitation():
    env=build_reference()
    assert env.current_reward()["score"]==1
    reordered=env.state.model_copy(update={"elements":tuple(reversed(env.state.elements))})
    assert evaluate(reordered)==env.current_reward()
    assert perturb(build_reference(),"neutral_image_color").current_reward()["score"]==1
    # Explicitly documents an unresolved construct-validity limit.
    assert perturb(build_reference(),"irrelevant_decoration").current_reward()["score"]==1


def test_contrast_reference_pairs():
    assert contrast_ratio("#000000","#FFFFFF")==pytest.approx(21)
    assert contrast_ratio("#123456","#123456")==pytest.approx(1)
    assert contrast_ratio("#FFFFFF","#FFFF00")==pytest.approx(1.0738392309)


def test_actual_background_and_layer_order():
    env=build_reference()
    # Transparent headline now lies above a dark background of exactly its own text color.
    apply(env,{"op":"add_element","element":{"type":"shape","color":"#172033","x":100,"y":72,"width":600,"height":90,"z_index":-1}})
    assert env.current_reward()["elements"]["headline"]["contrast"]==pytest.approx(1)
    assert env.current_reward()["score"]<0


def test_task_changes_and_environment_isolation():
    a,b=build_reference(),build_reference(task=TaskSpec(headline="Winter Sale"))
    assert a.current_reward()["success"] and b.current_reward()["success"]
    assert evaluate(a.state.model_copy(update={"task":b.state.task}))["score"]<0
    apply(a,{"op":"delete_element","id":"e3"})
    assert b.current_reward()["success"]
    assert build_reference(task=TaskSpec(alignment="left")).current_reward()["success"]


def test_element_cap_and_ids_are_not_reused():
    env=MarketCanvasEnv(max_steps=100)
    for _ in range(env.state.max_elements):
        apply(env,{"op":"add_element","element":{"type":"text"}})
    assert env.step({"op":"add_element","element":{"type":"text"}})[4]["error"]
    apply(env,{"op":"delete_element","id":"e1"})
    apply(env,{"op":"add_element","element":{"type":"text"}})
    assert env.state.elements[-1].id=="e25"


def test_gymnasium_contract_and_semantic_parity():
    check_env(GymCanvasEnv(),skip_render_check=True)
    gym=GymCanvasEnv()
    obs,_=gym.reset(seed=11)
    assert gym.observation_space.contains(obs)
    action=json.dumps({"op":"submit"})
    assert gym.action_space.contains(action)
    assert gym.step(action)[1:4]==(-1,True,False)
