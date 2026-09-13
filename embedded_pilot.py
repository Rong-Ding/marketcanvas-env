"""New stimuli with embedded same-side and reversed-side repeat checks."""
import argparse
import copy
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import random
import shutil

from marketcanvas import MarketCanvasEnv, TaskSpec
from marketcanvas.scenarios import apply
from marketcanvas.viewer import evaluate_viewer, extract_regions
from viewer_pilot import ROOT, FAMILIES, DEFAULT_OUTPUT, dump, sha, canonical_hash, verify_study, settings
from viewer_analysis import analyze
import rating_server

STUDY = ROOT / 'results/viewer-embedded-v1'


def scene(family, seed, alternative):
    task = TaskSpec(headline=('Autumn Edit', 'Member Offers', 'New Season')[seed],
                    cta=('Browse now', 'See offers', 'Discover')[seed])
    env = MarketCanvasEnv(task=task)
    env.reset(seed=seed+30)
    width = (336 if alternative else 216) if family == 'cta_width' else 272
    image = dict(type='image', color='#DBEAFE', content='Product placeholder',
                 x=120, y=225, width=560, height=150)
    if family == 'image_contrast':
        image['color'] = '#172033' if alternative else '#DBEAFE'
    if family == 'image_position':
        image.update(x=300 if alternative else 80, width=200)
    specs = [dict(type='text', role='headline', content=task.headline, x=100, y=65+7*seed,
                  width=600, height=90, font_size=44), image,
             dict(type='shape', role='cta', content=task.cta, color='#FFFF00',
                  x=(800-width)//2, y=435+12*seed, width=width, height=72, font_size=28)]
    if family == 'competing_text_size':
        specs.append(dict(type='text', content='This week only', x=100, y=160, width=600,
                          height=60, font_size=38 if alternative else 20))
    for spec in specs:
        apply(env, {'op':'add_element', 'element':spec})
    env.begin_prepared_episode()
    if not env.current_reward()['success']:
        raise ValueError(f'Invalid planned stimulus {family}/{seed}/{alternative}')
    return env


def schedule():
    rng = random.Random(2026091401)
    cases = [(f, s) for f in FAMILIES for s in range(3)]
    while True:
        rng.shuffle(cases)
        if len({f for f, _ in cases[:6]}) == 4:
            break
    swaps = [False]*6+[True]*6
    rng.shuffle(swaps)
    repeat_types = [False]*3+[True]*3
    rng.shuffle(repeat_types)
    pending = list(range(6))
    positions, rows, main_index = {}, [], 0
    for pos in range(18):
        if pos in (7, 9, 11, 13, 15, 17):
            eligible = [i for i in pending if pos-positions[i] >= 4]
            i = rng.choice(eligible)
            pending.remove(i)
            rows.append(dict(case=cases[i], primary_index=i, repeat=True,
                             reversed=repeat_types[i], swap=swaps[i] ^ repeat_types[i]))
        else:
            i = main_index; main_index += 1; positions[i] = pos
            rows.append(dict(case=cases[i], primary_index=i, repeat=False,
                             reversed=False, swap=swaps[i]))
    return rows


def build(out=STUDY):
    out = Path(out)
    if out.exists():
        return verify_study(out)
    base = verify_study(DEFAULT_OUTPUT)
    out.mkdir(parents=True)
    (out/'stimuli').mkdir(); (out/'states').mkdir()
    shutil.copytree(DEFAULT_OUTPUT/'frozen_source', out/'frozen_source')
    shutil.copyfile(__file__, out/'frozen_source/embedded_pilot.py')
    public, predictions, key, primary_ids = [], {}, {}, {}
    # The schedule and all stimulus parameters are fixed before evaluating scores.
    for index, spec in enumerate(schedule(), 1):
        item = f'I{index:02}'
        family, seed = spec['case']
        pair = {'id':item}
        prediction = dict(family=family, fixture_seed=seed, swapped=spec['swap'])
        if not spec['repeat']:
            primary_ids[spec['primary_index']] = item
        source_id = primary_ids[spec['primary_index']]
        for side, alternative in [('left',spec['swap']),('right',not spec['swap'])]:
            image_name = f'stimuli/{item}_{side}.png'
            pair[side] = image_name
            if spec['repeat']:
                source_side = ('right' if side=='left' else 'left') if spec['reversed'] else side
                shutil.copyfile(out/f'stimuli/{source_id}_{source_side}.png', out/image_name)
                shutil.copyfile(out/f'states/{source_id}_{source_side}.json', out/f'states/{item}_{side}.json')
                prediction[side] = copy.deepcopy(predictions[source_id][source_side])
                first = next(p for p in public if p['id']==source_id)
                pair.update(headline=first['headline'], cta=first['cta'])
            else:
                env = scene(family,seed,alternative)
                env.save_png(out/image_name)
                dump(out/f'states/{item}_{side}.json',env.export_trajectory())
                regions = extract_regions(env.state)
                primary = evaluate_viewer(env.state, regions=regions)
                sensitivity = {}
                for name, config in settings().items():
                    score = evaluate_viewer(env.state, config, regions=regions)
                    sensitivity[name] = {k:score[k] for k in ('score','D1','D2','D3')}
                prediction[side] = dict(alternative=alternative, primary=primary, sensitivity=sensitivity)
                pair.update(headline=env.state.task.headline, cta=env.state.task.cta)
        predictions[item] = prediction
        public.append(pair)
        key[item] = dict(primary_id=source_id, kind=('reversed' if spec['reversed'] else 'same') if spec['repeat'] else 'primary')
    dump(out/'public_manifest.json',dict(study_id='viewer-embedded-v1', pairs=public,
         reviewer_count=1, questions=['discoverability','preference'], answers=['left','right','tie','unsure']))
    dump(out/'predictions.json',predictions)
    dump(out/'repeat_key.json',key)
    ui = out/'frozen_source/marketcanvas/web/rating.html'
    text = ui.read_text().replace('How do these designs read?', 'Design review: new batch')
    text = text.replace('A short pilot about finding an action button and your overall design preference.',
                        '18 comparisons about finding an action button and your overall design preference. Your earlier responses are saved separately.')
    text = text.replace("a.download='viewer-pilot-ratings.json'", "a.download=manifest.study_id+'-ratings.json'")
    text = text.replace("$('results').append(h,p,table,note)", """$('results').append(h,p,table,note);if(result.repeat_checks){const rh=document.createElement('h2');rh.textContent='Repeat consistency (reported separately)';$('results').append(rh);for(const [q,types] of Object.entries(result.repeat_checks)){for(const [kind,c] of Object.entries(types)){const line=document.createElement('p');line.textContent=`${q} · ${kind} sides: ${c.consistent}/${c.eligible} consistent image choices; ${c.unsure} unsure; ${c.same_displayed_side}/${c.directional} directional answers kept the displayed side.`;$('results').append(line)}}}""")
    ui.write_text(text)
    shutil.copyfile(ROOT/'docs/EMBEDDED_PILOT.md',out/'METHOD.md')
    protocol = copy.deepcopy(base)
    protocol.update(study_id='viewer-embedded-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        design='post-feedback exploratory new batch with embedded checks; one reviewer',
        pair_count=18, unique_pairs=12, repeat_count=6, same_side_repeats=3, reversed_side_repeats=3,
        assignment_seed=2026091401, minimum_repeat_index_gap=4,
        analysis_note='Primary agreement uses 12 first presentations only. See METHOD.md.',
        files_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    dump(out/'protocol.json',protocol)
    (out/'freeze.sha256').write_text(canonical_hash(protocol)+'\n')
    return verify_study(out)


def report(study, record, destination):
    result = analyze(study,record)  # Completion gate applies to all 18 presentations.
    key = json.loads((Path(study)/'repeat_key.json').read_text())
    first = [r for r in result['pairs'] if key[r['pair']]['kind']=='primary']
    for summary in result['summaries']:
        q, model = summary['question'],summary['model']
        eligible = [r for r in first if r[q]!='unsure']
        directional = [r for r in eligible if r[q] in ('left','right')]
        decisive = [r for r in directional if r[model]!='tie']
        matches = sum(r[q]==r[model] for r in eligible)
        summary.update(agreements=matches,rated_non_unsure=len(eligible),
            agreement_fraction=matches/len(eligible) if eligible else None,
            human_ties=len(eligible)-len(directional),unsure=len(first)-len(eligible),
            model_ties=sum(r[model]=='tie' for r in eligible),
            decisive_on_human_directional=len(decisive),human_directional=len(directional),
            directional_agreements=sum(r[q]==r[model] for r in decisive))
    checks = {}
    for q in ('discoverability','preference'):
        checks[q] = {}
        for kind in ('same','reversed'):
            counts = dict(total=0,eligible=0,consistent=0,unsure=0,directional=0,same_displayed_side=0)
            for item,m in key.items():
                if m['kind']!=kind: continue
                a,b = record['ratings'][m['primary_id']][q],record['ratings'][item][q]
                counts['total']+=1
                if 'unsure' in (a,b): counts['unsure']+=1; continue
                canonical = {'left':'right','right':'left'}.get(b,b) if kind=='reversed' else b
                counts['eligible']+=1; counts['consistent']+=a==canonical
                if a in ('left','right') and b in ('left','right'):
                    counts['directional']+=1; counts['same_displayed_side']+=a==b
            checks[q][kind]=counts
    result.update(pair_count=12,presentation_count=18,repeat_checks=checks,
        interpretation='Exploratory single-reviewer batch after earlier feedback. Reward agreement uses only 12 first presentations; six repeat checks are reported separately. Three checks per type cannot establish a causal side bias.')
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    dump(destination/'analysis.json',result)
    lines=['# Embedded pilot results','',result['interpretation'],'',
           '| Question | Reward | Agreement |','|---|---|---:|']
    for s in result['summaries']:
        if s['model'] in ('original','viewer'):
            lines.append(f"| {s['question']} | {s['model']} | {s['agreements']}/{s['rated_non_unsure']} |")
    lines.extend(['','## Repeat checks','',json.dumps(checks,indent=2),''])
    (destination/'REPORT.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve',action='store_true')
    parser.add_argument('--port',type=int,default=8768)
    args=parser.parse_args()
    build()
    if args.serve:
        if sha(Path(__file__))!=sha(STUDY/'frozen_source/embedded_pilot.py'):
            raise ValueError('Embedded analysis code differs from frozen study')
        for name in ('rating_server.py','viewer_analysis.py'):
            if sha(ROOT/name)!=sha(STUDY/'frozen_source'/name):
                raise ValueError(f'{name} differs from frozen study')
        rating_server.write_report=report
        store=rating_server.RatingStore(STUDY,ROOT/'private/viewer-embedded-v1/ratings.json')
        server=ThreadingHTTPServer(('127.0.0.1',args.port),rating_server.handler_for(store))
        print(f'Embedded pilot: http://127.0.0.1:{args.port}/',flush=True)
        server.serve_forever()
