import json
import numpy as np
import pytest
from marketcanvas.scenarios import CASES, load_scenario, build_reference, apply
from replay import replay


@pytest.mark.parametrize("case", CASES)
def test_loaded_scenes_get_full_budget_and_replay_without_edits(case, tmp_path):
    env=load_scenario(case)
    assert env.state.steps==0
    assert env.observe()["remaining_steps"]==40
    assert env.trajectory==[]
    assert len(env.setup_actions)==(3 if case=="reference" else 4)
    path=tmp_path/'trajectory.json'
    path.write_text(json.dumps(env.export_trajectory()))
    recovered=replay(path)
    assert recovered.observe()==env.observe()
    assert np.array_equal(recovered.render(),env.render())


def test_first_edit_budget_and_terminal_export(tmp_path):
    env=load_scenario('low_contrast')
    apply(env,{'op':'update_element','id':'e3','properties':{'text_color':'#172033'}})
    assert env.state.steps==1 and env.observe()['remaining_steps']==39
    result=apply(env,{'op':'submit'})
    assert env.state.steps==2 and result[1]==1
    path=tmp_path/'trajectory.json'
    path.write_text(json.dumps(env.export_trajectory()))
    assert replay(path).export_trajectory()==env.export_trajectory()


def test_setup_does_not_shorten_episode():
    env=load_scenario()
    for _ in range(39):
        assert not env.step({'op':'invalid'})[2]
    assert env.step({'op':'submit'})[1:4]==(1,True,False)


def test_original_export_formats_still_replay(tmp_path):
    env=build_reference()
    apply(env,{'op':'submit'})
    for data in (json.dumps(env.trajectory),'\n'.join(json.dumps(t) for t in env.trajectory)):
        path=tmp_path/'legacy.json'
        path.write_text(data)
        assert replay(path).observe()==env.observe()
