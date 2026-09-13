# Reproducing the research results

The main submission is ../WRITEUP.md. The full account is ../docs/RESEARCH_PAPER.md.
This directory releases the minimum single-reviewer labels required to reproduce
those results. It contains no free-text notes, reviewer identity or account data.
The reviewer helped design the study. The second batch followed discussion of
initial results; it is exploratory and not an independent blinded validation.

From the project root, after installing the locked dependencies:

```sh
python reproduce_research.py
```

The command verifies both frozen studies, recomputes their human-agreement
reports, and refits the contextual reward from 24 first presentations. It writes
`research/reproduced/`. A repeat run requires a fresh `--output` directory.
No external service or private Downloads file is required. To collect new
annotations use `embedded_pilot.py --serve`, which saves separately under private/.
Do not overwrite the released labels with new responses.

Expected results:

- Initial viewer: 7/12 discoverability; 3/12 preference.
- Embedded viewer, first presentations only: 4/12 for each outcome.
- Embedded consistency: discoverability same-side 3/3, reversed 1/3;
  preference same-side 2/3, reversed 1/3.
- Fitted reward: 10/24 training, 8/24 grouped cross-validation agreement;
  frozen viewer on the same 24 pairs: 11/24.

Original and embedded labels retain study IDs, collection-completion timestamp,
original freeze fingerprint and the two categorical answers per item. Per-answer
timestamps, revision history and notes have been removed. Predictions and repeat
mapping come from the immutable study packs. Those packs retain pre-annotation
source/doc snapshots; 'pending' wording inside them is historical provenance.
The fresh-install audit is documented in ../docs/VERIFICATION.md.
