"""Fit current human discoverability annotations; repeats are never training rows."""
from pathlib import Path
import json
import argparse
from datetime import datetime,timezone
import numpy as np
from marketcanvas.models import State
from marketcanvas.fitted_reward import FEATURES, fit, predict, evaluate_fitted
from viewer_pilot import ROOT, verify_study, sha, dump
from marketcanvas.fitted_reward import features

DEFAULT=ROOT/'private/fitted-reward-v1'


def load_rows(ratings_dir=None):
    rows=[]; sources=[]
    for name in ('viewer-pilot-v1','viewer-embedded-v1'):
        study=ROOT/'results'/name
        verify_study(study)
        path=(Path(ratings_dir)/(name+'.json')) if ratings_dir else ROOT/'private'/name/'ratings.json'
        if not path.exists() and ratings_dir is None:
            path=ROOT/'research/ratings'/(name+'.json')
        record=json.loads(path.read_text())
        manifest=json.loads((study/'public_manifest.json').read_text())
        predictions=json.loads((study/'predictions.json').read_text())
        if record['freeze_hash']!=(study/'freeze.sha256').read_text().strip() or not record['completed_at']:
            raise ValueError('Completed, matching human ratings required')
        if set(record['ratings'])!={p['id'] for p in manifest['pairs']}:
            raise ValueError('Rating IDs differ from the frozen study')
        key=json.loads((study/'repeat_key.json').read_text()) if name=='viewer-embedded-v1' else None
        sources.append(dict(study=name,freeze=record['freeze_hash'],ratings_sha256=sha(path)))
        for pair in manifest['pairs']:
            item=pair['id']; rating=record['ratings'][item]
            if key and key[item]['kind']!='primary': continue
            if rating['discoverability']=='unsure': continue
            states=[]
            for side in ('left','right'):
                export=json.loads((study/f'states/{item}_{side}.json').read_text())
                assert not export['transitions']
                obs=export['initial_observation']
                states.append(State.model_validate({k:v for k,v in obs.items() if k in State.model_fields}))
            p=predictions[item]
            rows.append(dict(id=f'{name}/{item}',study=name,family=p['family'],group=p['fixture_seed'],
                 human=rating['discoverability'],preference=rating['preference'],
                 left=features(states[0]).tolist(),right=features(states[1]).tolist(),
                 frozen_viewer=('left' if p['left']['primary']['score']>p['right']['primary']['score'] else 'right' if p['left']['primary']['score']<p['right']['primary']['score'] else 'tie'),
                 states=states))
    return rows,sources


def summarize(rows,predictions):
    return dict(n=len(rows),agreements=sum(r['human']==p['choice'] for r,p in zip(rows,predictions)),
        log_loss=float(np.mean([-np.log(max(p['probabilities'][r['human']],1e-300)) for r,p in zip(rows,predictions)])),
        predicted_ties=sum(p['choice']=='tie' for p in predictions),
        human_ties=sum(r['human']=='tie' for r in rows))


def run(out=DEFAULT, ratings_dir=None):
    out=Path(out)
    if out.exists(): raise ValueError('Output exists; choose a fresh --output to preserve previous fits')
    rows,sources=load_rows(ratings_dir)
    out.mkdir(parents=True)
    # Record choices before optimizing, but this is explicitly post-data exploratory.
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),target='discoverability only',
        inputs=sources,unique_pairs=len(rows),excluded='all six embedded repeats; preference labels not fitted',
        features=list(FEATURES),ridge=.1,alpha=.15,
        validation='Three folds grouped by fixture seed across both batches; all four families held together for each seed. Diagnostic only: hypotheses were developed after inspecting all ratings.',
        source_sha256={n:sha(ROOT/n) for n in ('fit_human_reward.py','marketcanvas/fitted_reward.py','docs/FITTED_REWARD.md')})
    dump(out/'FIT_PLAN.json',plan)
    left=np.array([r['left'] for r in rows]); right=np.array([r['right'] for r in rows])
    labels=[r['human'] for r in rows]
    model=fit(left,right,labels)
    full=[predict(r['left'],r['right'],model) for r in rows]
    oof=[None]*len(rows); folds=[]
    for group in sorted({r['group'] for r in rows}):
        train=[i for i,r in enumerate(rows) if r['group']!=group]
        test=[i for i,r in enumerate(rows) if r['group']==group]
        fold_model=fit(left[train],right[train],[labels[i] for i in train])
        for i in test: oof[i]=predict(left[i],right[i],fold_model)
        folds.append(dict(group=group,train_ids=[rows[i]['id'] for i in train],test_ids=[rows[i]['id'] for i in test],
                          model=fold_model,summary=summarize([rows[i] for i in test],[oof[i] for i in test])))
    result=dict(training=summarize(rows,full),grouped_cv=summarize(rows,oof),folds=folds,
        frozen_viewer_agreement=sum(r['human']==r['frozen_viewer'] for r in rows),
        original_always_tie_agreement=sum(r['human']=='tie' for r in rows),
        model=model,rows=[{**{k:v for k,v in r.items() if k!='states'},'fitted':p,'cross_validated':v,
                          'rewards':[evaluate_fitted(s,model) for s in r['states']]} for r,p,v in zip(rows,full,oof)])
    dump(out/'model.json',model); dump(out/'RESULTS.json',result)
    lines=['# Exploratory fitted discoverability reward','',
        'One reviewer; both existing batches reused. This is reward-model fitting, not LLM training. Six repeated presentations excluded.',
        '', '| Evaluation | Exact agreement | Mean three-outcome log loss |','|---|---:|---:|']
    for name,s in [('Training fit (optimistic)',result['training']),('Grouped cross-validation (diagnostic)',result['grouped_cv'])]:
        lines.append(f"| {name} | {s['agreements']}/{s['n']} | {s['log_loss']:.3f} |")
    lines += ['',f"Frozen viewer: {result['frozen_viewer_agreement']}/{len(rows)}. Original always-tie heuristic: {result['original_always_tie_agreement']}/{len(rows)}.",
        '', '## Fitted coefficients','', '| Feature | Standardized weight |','|---|---:|']
    for f,w in model['standardized_weights'].items(): lines.append(f'| {f} | {w:.4f} |')
    lines += ['', 'Unidentified from within-pair differences: '+', '.join(model['unidentified'])+'.',
        '',f"Fitted tie logit: {model['tie_logit']:.4f}. No left/right bias intercept is fitted.",
        '', 'Cross-validation refits all coefficients and scales inside each training fold. It cannot undo earlier hypothesis selection using these ratings. Do not call this independent validation or marketing-quality calibration.',
        '', 'The model remains opt-in and the current canvas reward is unchanged. Method, equations, limitations and future RL use: docs/FITTED_REWARD.md.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','folds')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--ratings-dir',type=Path)
    args=parser.parse_args()
    run(args.output,args.ratings_dir)
