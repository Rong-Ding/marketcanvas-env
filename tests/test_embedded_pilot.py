import json
from collections import Counter

import pytest

from embedded_pilot import STUDY, build, report
from rating_server import RatingStore
from viewer_pilot import sha, verify_study


def test_embedded_freeze_schedule_and_pixel_mapping():
    build()
    protocol=verify_study(STUDY)
    key=json.loads((STUDY/'repeat_key.json').read_text())
    predictions=json.loads((STUDY/'predictions.json').read_text())
    manifest=json.loads((STUDY/'public_manifest.json').read_text())
    assert Counter(m['kind'] for m in key.values()) == {'primary':12,'same':3,'reversed':3}
    assert len(manifest['pairs'])==18
    assert all(set(p)=={'id','left','right','headline','cta'} for p in manifest['pairs'])
    assert sum(predictions[i]['swapped'] for i,m in key.items() if m['kind']=='primary')==6
    for item,m in key.items():
        if m['kind']=='primary': continue
        assert int(item[1:])-int(m['primary_id'][1:])>=4
        for side in ('left','right'):
            other={'left':'right','right':'left'}[side] if m['kind']=='reversed' else side
            assert sha(STUDY/f'stimuli/{item}_{side}.png')==sha(STUDY/f"stimuli/{m['primary_id']}_{other}.png")
            assert predictions[item][side]==predictions[m['primary_id']][other]
    assert protocol['unique_pairs']==12


def test_results_gate_denominators_and_reversed_consistency(tmp_path):
    key=json.loads((STUDY/'repeat_key.json').read_text())
    store=RatingStore(STUDY,tmp_path/'synthetic.json')
    with pytest.raises(ValueError):
        report(STUDY,store.read(),tmp_path/'analysis')
    for index,(item,m) in enumerate(key.items()):
        answer='right' if m['kind']=='reversed' else 'left'
        store.save(dict(pair_id=item,discoverability=answer,preference='tie',notes='Synthetic QA only',revision=index))
    result=report(STUDY,store.finish(18),tmp_path/'analysis')
    assert all(s['rated_non_unsure']==12 for s in result['summaries'])
    assert result['pair_count']==12 and result['presentation_count']==18
    for q in ('discoverability','preference'):
        for kind in ('same','reversed'):
            assert result['repeat_checks'][q][kind]['consistent']==3
    assert result['repeat_checks']['discoverability']['same']['same_displayed_side']==3
    assert result['repeat_checks']['discoverability']['reversed']['same_displayed_side']==0
    record=store.read()
    reversed_item=next(i for i,m in key.items() if m['kind']=='reversed')
    record['ratings'][reversed_item]['discoverability']='unsure'
    result=report(STUDY,record,tmp_path/'with_unsure')
    assert result['repeat_checks']['discoverability']['reversed']['eligible']==2
    assert result['repeat_checks']['discoverability']['reversed']['unsure']==1
