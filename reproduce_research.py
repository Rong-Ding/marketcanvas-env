"""Reproduce released human results without any private local files."""
import argparse
import json
from pathlib import Path
from viewer_pilot import ROOT, verify_study
from viewer_analysis import write_report
from embedded_pilot import report
from fit_human_reward import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'research/reproduced')
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a new --output to preserve previous runs')
    released=ROOT/'research/ratings'
    for name,analyzer in [('viewer-pilot-v1',write_report),('viewer-embedded-v1',report)]:
        study=ROOT/'results'/name
        verify_study(study)
        r=json.loads((released/(name+'.json')).read_text())
        if r['freeze_hash']!=(study/'freeze.sha256').read_text().strip():
            raise ValueError('Released annotations do not match study fingerprint')
        result=analyzer(study,r,args.output/name)
        print(name,[(s['question'],s['agreements'],s['rated_non_unsure']) for s in result['summaries'] if s['model']=='viewer'])
    run(args.output/'fitted-reward',ratings_dir=released)


if __name__=='__main__':
    main()
