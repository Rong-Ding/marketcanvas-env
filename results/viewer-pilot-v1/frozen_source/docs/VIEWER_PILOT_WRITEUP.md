# From valid layouts to viewer-oriented evaluation

## Question and motivation

MarketCanvas's original reward checks explicit content, readability, contrast and simple alignment. It can assign equal reward to valid designs that differ in visual competition. We ask whether adding a small sequential viewer model improves agreement with a reviewer's judgments of CTA discoverability. Overall design preference is measured separately: finding a button easily need not imply preferring the composition.

Work on scanpath prediction for graphic documents motivates examining viewing sequences [1]. Psychological work distinguishes pleasure and interest [2], while cross-cultural aesthetic research shows that order, complexity and appreciation should not be treated as interchangeable [3]. These sources motivate our questions; they do not validate our model parameters or establish marketing effectiveness. The contribution is an explicit, testable measurement hypothesis rather than a claim of human-like vision.

## Model and reward

The viewer selects among visible element regions. Features come from the rendered canvas: visible pixel area, local luminance-edge contrast, and centroid. Semantic roles are excluded from its selection policy, although element segmentation is provided by the simulator. Transparent text contributes its visible ink instead of its empty bounding box. We model state as gaze position g and a set M of inspected regions. For region i:

`P(i|g,M,C) = softmax_i[beta*S_i - lambda*d(g,p_i) - rho*1(i in M)]`

Here S combines normalized square-root area and edge contrast equally; d is physical distance divided by the canvas diagonal. Primary settings are beta=2, lambda=1, rho=1 and an initial gaze at the canvas center. Revisits are allowed. This is a heuristic model of early visual exploration, not a trained eye-movement model or a semantic searcher. Exact probability propagation yields D_K, the probability the CTA is inspected within K=1,2,3 model glances. No gaze durations or milliseconds are simulated.

The primary extension combines D_2 with original quality: `q_v=.85*q_original_before_cap+.15*D_2`. It retains the essential-validity cap and maps quality to reward [-1,1]. Task success stays unchanged. A future trainer could use this as a terminal reward; an evaluator can model temporal viewing without paying reward on every editing action. The current playground and MCP retain the original reward, and no LLM is trained in this study.

## Pilot design

Twelve pairs cross four planned manipulations with three campaign variants: image contrast, competing-text size, CTA width, and image position. Both members meet the same task contract and receive original reward 1.0. Pair generation does not select for agreement with the viewer model. Pair order is shuffled and alternative placement balanced across sides using a fixed seed. We freeze stimuli, predictions, parameters, source snapshots, dependencies and the analysis specification before rating.

One reviewer, involved in developing the hypothesis, answers two questions per pair: which CTA is easier to find, and which design is preferred overall. Left, right, about equal and unsure are allowed. Scores and manipulation names remain hidden until explicit completion; saved ratings then lock before predictions are revealed. This is a single-reviewer development pilot, not independent blinded validation. It measures perceived discoverability and preference, not gaze, response time, comprehension or conversion.

## Analysis and current status

**Human results are pending until the reviewer completes the pilot.** No ratings or agreement values are imputed. The completed analysis will be generated from the local response record and frozen predictions.

For each judgment separately, report exact pairwise agreement, model/human ties, unsure counts and directional coverage. A model tie agrees only with a human tie; unsure responses are excluded from the denominator. Because the baseline ties all valid pairs, report its lack of discrimination explicitly. Inspect individual disagreements rather than only aggregate agreement. Sensitivity analysis changes prominence, distance, revisit penalty and starting position one factor at a time; K=1/3 and reward weights .05/.15/.30 are secondary. Positive weight changes preserve rankings within these equally valid pairs, but change score magnitudes.

Implementation checks verify probability mass against a closed-form uniform-viewer case, role blindness, occlusion behavior, deterministic evaluation, persistence, score hiding, and completion locking. Human ratings remain separate from automated QA data.

## Limits and interpretation

The model may favor a large CTA or overlook semantic cues that make a smaller one recognizable. Its region attribution, contrast normalization and exploration bias are assumptions. The reviewer deliberately searches for a named CTA while the model has no semantics; disagreement may reveal this mismatch. A distracting message can reduce early discovery yet increase interest, which is why preference is a separate outcome.

The twelve related pairs do not constitute twelve independent participants. No population-level significance, trained-agent improvement, or gaze-model validation is claimed. A favorable pilot would justify a frozen follow-up with new stimuli and independent reviewers; an unfavorable result would identify which assumptions or constructs need revision. Any changes made after examining ratings must be labeled exploratory.

## References

1. [Chakraborty et al. Predicting Visual Attention in Graphic Design Documents](https://arxiv.org/abs/2407.02439).
2. [Graf & Landwehr. Aesthetic Pleasure versus Aesthetic Interest](https://pmc.ncbi.nlm.nih.gov/articles/PMC5276863/).
3. [Van Geert, Ding & Wagemans. A Cross-Cultural Comparison of Aesthetic Preferences for Neatly Organized Compositions](https://journals.sagepub.com/doi/abs/10.1177/02762374241245917). Online 2024; issue 2025.

Reproduce with `python viewer_pilot.py`, then `python rating_server.py`. See `docs/VIEWER_METHOD.md` for feature definitions, implementation and analysis details.
