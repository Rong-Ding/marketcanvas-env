import json
import numpy as np
import pytest

from embedded_pilot import scene
from marketcanvas.fitted_reward import features,fit,predict,evaluate_fitted
from marketcanvas.scenarios import apply
from fit_human_reward import load_rows,DEFAULT


def test_object_center_distance_and_width_independence():
    centered=scene('image_position',0,True)
    shifted=scene('image_position',0,False)
    assert features(shifted.state)[2]>features(centered.state)[2]
    narrow=scene('cta_width',0,False)
    wide=scene('cta_width',0,True)
    assert features(wide.state)[0]>features(narrow.state)[0]
    assert features(wide.state)[2]==features(narrow.state)[2]


def test_synthetic_fit_side_equivariance_and_unidentified_effects():
    left=np.zeros((12,4)); right=np.zeros_like(left)
    left[:6,0]=1;right[6:,0]=1
    labels=['left']*6+['right']*6
    m=fit(left,right,labels)
    assert m['weights'][0]>0
    assert len(m['unidentified'])==3
    a=predict(left[0],right[0],m);b=predict(right[0],left[0],m)
    assert a['choice']=='left' and b['choice']=='right'
    assert a['probabilities']['left']==pytest.approx(b['probabilities']['right'])
    assert sum(a['probabilities'].values())==pytest.approx(1.)
    tied=fit(left,right,['tie']*12)
    assert predict(left[0],right[0],tied)['choice']=='tie'


def test_reward_validity_cap_and_unsupported_geometry():
    narrow=features(scene('cta_width',0,False).state)
    wide=features(scene('cta_width',0,True).state)
    m=fit(np.array([wide]*4),np.array([narrow]*4),['left']*4)
    env=scene('cta_width',0,True)
    r=evaluate_fitted(env.state,m)
    assert -1<=r['score']<=1 and r['supported']
    apply(env,dict(op='update_element',id='e3',properties={'text_color':'#FFFF00'}))
    assert evaluate_fitted(env.state,m)['score']<=-.02
    apply(env,dict(op='delete_element',id='e3'))
    assert evaluate_fitted(env.state,m)['supported'] is False


@pytest.mark.skipif(not (DEFAULT/'RESULTS.json').exists(),reason='Optional private human-data integration check')
def test_dataset_and_fold_separation():
    rows,_=load_rows()
    assert len(rows)==24 and len({r['id'] for r in rows})==24
    result=json.loads((DEFAULT/'RESULTS.json').read_text())
    all_test=[]
    for fold in result['folds']:
        assert not set(fold['train_ids']) & set(fold['test_ids'])
        train=[r for r in rows if r['id'] in fold['train_ids']]
        test=[r for r in rows if r['id'] in fold['test_ids']]
        assert len(train)==16 and len(test)==8
        assert not {r['group'] for r in train} & {r['group'] for r in test}
        differences=np.array([np.array(r['left'])-r['right'] for r in train])
        assert np.allclose(fold['model']['scale'],np.sqrt(np.mean(differences**2,axis=0)))
        all_test.extend(fold['test_ids'])
    assert sorted(all_test)==sorted(r['id'] for r in rows)
