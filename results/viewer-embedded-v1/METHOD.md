# New batch with embedded consistency checks

This exploratory follow-up uses 12 new stimulus pairs (four manipulation families,
three campaign/layout variants each), plus six repetitions: three same-side and
three reversed-side. New stimuli change campaign text and geometry; they remain
closely related to the original stimulus templates and are not independent design
families. The reviewer has seen the original model results and knows checks exist.
This is not a blinded confirmation study.

Stimulus parameters, assignment seed 2026091401, and the unchanged viewer model
are fixed before collecting these responses. No human answers are read by the
generator. Stimuli are not selected by model agreement or score differences.
Twelve first presentations have balanced condition orientation (six each). Six
repeat sources are drawn by the deterministic randomized schedule, span all four
manipulation families, and appear at least four item positions after their first
presentation. Repeats are interleaved with new pairs, using neutral item IDs.
Their identities, mappings and scores are inaccessible through the rating API
until completion (mapping itself is local analysis material, never an API route).

Primary analysis: exact reward/human agreement for each question, using only the
12 first presentations. Report ties, uncertainty and prediction coverage. Never
treat 18 presentations as 18 independent designs. Preserve the original round
and abandoned reversed-only follow-up as separate records.

Repeat analysis: compare image identity choices separately for the three same-side
and three reversed-side checks. Map left/right back before scoring reversed
consistency; tie agrees only with tie. Exclude unsure with counts. Also count
repeated displayed-side choices among pairs with two directional answers.
Same-side stability provides context for reversed-side stability, but different
items are used for each type and three checks per type are insufficient to
attribute inconsistency causally to position. Memory, feedback and changing
judgments remain possible explanations. Do not discard inconsistent answers,
select the answer favoring the model, or average all repeats into extra evidence.

Responses save locally. Completing all 18 items locks them and reveals primary
reward agreement and separately labeled repeat statistics. No reward fitting,
external LLM judging, company assets, or external account access is involved.
