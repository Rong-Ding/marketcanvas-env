# Viewer pilot: method and implementation

## Scope and status

This is a single-reviewer development pilot for MarketCanvas-Env. It compares a static task-validity reward with an extension predicting early CTA inspection. The reviewer helped formulate the hypothesis. Numerical predictions and manipulation names are withheld in the rating interface until completion, but the study is not an independent blinded validation. There is no eye tracking, timed visual-search measurement, comprehension test, trained gaze model, LLM experiment, or RL fine-tuning in this pilot.

The original playground and MCP continue to deliver the original terminal reward. `evaluate_viewer(state)` is a pure alternative evaluator that could supply a versioned terminal reward to a future rollout runner. Reading a preview never pays a reward. A viewer's predicted sequence and the designer's editing sequence are different objects.

## Research question

Among banners that satisfy the same explicit task requirements, does an early-viewing model agree better than the original reward with the reviewer's judgments of CTA discoverability? Overall design preference is recorded as a distinct secondary outcome. Aesthetic preference need not favor immediate discovery, and attention does not establish comprehension or marketing effectiveness.

## Literature and adaptation

- [Chakraborty et al., Predicting Visual Attention in Graphic Design Documents](https://arxiv.org/abs/2407.02439) model spatial attention and fixation sequences using inverse RL. This motivates investigating sequential viewing. Our implementation does not reproduce their model, learn a policy from gaze data, or transfer their validated performance to these stimuli.
- [Graf and Landwehr, Aesthetic Pleasure versus Aesthetic Interest](https://pmc.ncbi.nlm.nih.gov/articles/PMC5276863/) investigate distinct processing routes to aesthetic responses. This motivates separating discoverability from overall preference; neither is relabeled as a direct measurement of pleasure or fascination.
- [Van Geert, Ding and Wagemans, A Cross-Cultural Comparison of Aesthetic Preferences for Neatly Organized Compositions](https://journals.sagepub.com/doi/abs/10.1177/02762374241245917) report shared and differing relationships involving order, complexity and appreciation in their sampled groups. First online 2024, journal issue 2025. We use this as a reason to avoid equating one visual feature with universal quality, not as a source of demographic viewer parameters.

## Formal viewer

The canvas C is fixed during viewing. At model glance t, viewer state is `(g_t, M_t, t)`: gaze location, set of inspected regions, and glance count. The model has no language understanding or search-goal representation. It approximates early visual exploration; the human question explicitly names a CTA, creating a deliberate construct mismatch to examine.

Each candidate region has a stable ID, visible area A_i, contrast C_i, and position p_i. Semantic roles and task text are not supplied to the transition function. Simulator-provided element segmentation remains privileged information: this is not an end-to-end screenshot perception model. Shape and button label are one region; transparent text contributes its rendered ink.

Prominence:

`S_i = w_A * sqrt(A_i)/max_j sqrt(A_j) + (1-w_A) * C_i/max_j C_j`

Selection probabilities:

`P(i | g_t, M_t, C) = softmax_i[beta*S_i - lambda*d(g_t,p_i) - rho*1(i in M_t)]`

After selecting i, gaze moves to its centroid and i is added to the inspected set. Repeated selections are allowed and consume a glance. The revisit penalty is a heuristic exploration bias, not a calibrated account of inhibition of return or memory physiology. Gaze coordinates are normalized; distance is Euclidean physical pixel distance divided by the 800x600 canvas diagonal. The initial center position is not itself counted as inspecting an element.

Primary parameters: `w_A=.5`, `beta=2`, `lambda=1`, `rho=1`, initial gaze `(0.5,0.5)`. These are explicit engineering hypotheses. The model propagates probability mass exactly over `(current_region, visited_bitmask)` states for K=1,2,3. No Monte Carlo sampling, time durations, or random human behavior are claimed. The dynamic program reports visit probabilities for every region without knowing which is the CTA. Only the evaluator subsequently identifies the unique CTA to extract D_K = probability it has been inspected within K model glances. Missing/nonunique CTA yields D_K=0 in the evaluator.

## Pixel feature extraction

`marketcanvas/viewer.py::extract_regions` renders the complete scene using the existing Pillow renderer, then renders it once per element with that element removed. A region's visible support is the set of RGB pixels that differ from the full rendering. An entirely occluded element or one with no visible effect has no region. Area is the support's pixel count, not the editable bounding-box area; centroid is the support's mean coordinate.

The full frame is converted to linearized sRGB luminance using the shared renderer utility. At each pixel, edge strength is the maximum absolute luminance difference to its four immediate neighbors. A region's contrast is the mean of positive edge strengths on its visible support (zero if none). This is an appearance heuristic, distinct from the WCAG foreground/background contrast check retained by the original reward. It is sensitive to antialiasing, segmentation and local edges. Overlapping identical objects and edits to a pixel's underlying layer can expose limitations of the removal-based attribution.

Square-root area and mean edge contrast are each normalized by their maximum across regions. Consequently an edit can alter other regions' normalized prominence as well as its own. Feature extraction uses repeated rendering and is appropriate for the tiny pilot, not 10,000 parallel rollout infrastructure.

## Reward and sensitivity

Let q_0 be the original weighted quality before its validity cap. The extension uses `q_v=(1-alpha)*q_0+alpha*D_2` with alpha=.15, reapplies the original essential-failure cap q<=.49, and returns `2*q_v-1`. Task success is unchanged. Alpha=0 recovers the original reward. All 24 pilot scenes must already score 1.0 under the original evaluator; its pairwise ties make this a study of added discrimination within valid designs.

Frozen one-factor sensitivity settings: beta=1 or 4; lambda=0 or 2; rho=0 or 2; initial gaze `(0.5,0.25)`, alongside the primary configuration. D_1/D_3 are secondary viewing budgets. Alpha=.05/.15/.30 is also recorded. For these equally valid pairs, any positive alpha preserves the D_2 ranking; weight variation changes the score magnitude, not pair ordering. No settings are selected using the reviewer's responses.

## Stimulus design and presentation

There are four manipulation families, each instantiated with three campaign/headline/button-label variants: 12 pairs, 24 banners. Families are image-placeholder contrast, competing decorative-text size, CTA width, and image position. In each pair, the declared intervention changes while the remaining scene specifications stay fixed; changing a feature can legitimately affect several model terms downstream. Semantic content stays matched within each pair. No pair is filtered out for failing to favor the proposed model.

Pair order and left/right assignment are fixed with seed 20260913; six pairs show the alternative on each side. All scenes are generated through the public action API. Setup is saved in replayable exports outside the participant's edit budget. The evaluator's existing constraints are checked before a stimulus is accepted. These are co-designed development fixtures, not a random population sample. Repeated themes and locations can influence judgment; randomized order does not eliminate such effects.

## Review protocol and persistence

The independent rating server runs at localhost:8766. It shows both banners, their shared campaign and CTA label, and two questions: which button is easier to find, and which design is preferred overall. Each permits left/right/about equal/unsure. Optional notes are at most 1,000 characters. The reviewer may view the image at full size. Choices save on selection; notes save when leaving the field or using navigation/save controls. Saving partial answers is allowed. Reloading resumes the first incomplete pair.

Responses live in `private/viewer-pilot-v1/ratings.json`, excluded from Git by `.gitignore`. JSON replacement is atomic. An optimistic revision check prevents a second tab from silently overwriting newer responses. The server accepts same-origin JSON mutations and serves only the public manifest, permitted images, rating interface, and the reviewer's record. Predictions, scene definitions, protocol details and source snapshots are not served during rating. Local filesystem access can still reveal them; this is procedural score masking, not an adversarial concealment mechanism.

After every pair has both responses, the reviewer explicitly chooses **Finish pilot and reveal comparison**. This locks ratings and reveals analysis. Repeated completion is idempotent. Scores are not shown simply because the last radio button was selected. No automated QA responses are written into the real participant record.

## Freeze and reproducibility

`viewer_pilot.py` creates `results/viewer-pilot-v1/` once. It stores the public manifest, rendered stimuli, replayable scenes, predictions, protocol, sensitivity settings, dependency versions, font hash and the relevant model/server/UI/analysis source snapshots. `freeze.sha256` fingerprints the protocol, which fingerprints all generated files. Rerunning verifies the pack rather than overwriting it. The rating server verifies the freeze and refuses to run altered analysis/server code against it. Responses reference the fingerprint. New hypotheses or implementation changes require a separately identified study version and separate response file.

## Analysis plan

The primary comparison is exact pairwise agreement with perceived discoverability; preference is analyzed separately. Score differences <=1e-6 are model ties. A reviewer tie agrees only with a model tie. Unsure responses are excluded from the agreement denominator and counted. Report model ties, human ties, and directional coverage explicitly; never quietly award a model tie half credit. Report each pair and one-factor sensitivity results, including disagreements.

With one reviewer involved in hypothesis development, there are no population significance claims, no train/test generalization estimate and no eye-movement validation. Twelve pairs are not twelve independent participants. The analysis is descriptive evidence for refining the measurement. Any post-rating modifications must be labeled exploratory and tested on new material. No completion or human results are inferred from model predictions.

## Implementation map and verification

| File | Responsibility |
|---|---|
| `marketcanvas/viewer.py` | Pixel features, role-blind sequential model, alternative evaluator |
| `viewer_pilot.py` | Planned fixtures, predictions, sensitivity, source/data freeze |
| `rating_server.py` | Local review API, persistent answers, completion gate |
| `marketcanvas/web/rating.html` | Score-hidden paired review UI |
| `viewer_analysis.py` | Frozen-rule agreement analysis and local report |
| `tests/test_viewer_pilot.py` | Closed-form probability, role blindness, visibility, lifecycle, freeze and HTTP tests |

Tests check a uniform viewer against `P(visited by K)=1-(1-1/n)^K`, total probability mass, deterministic read-only evaluation, role-label invariance, exclusion of hidden elements, essential-invalidity penalties, and alpha-zero equivalence. Workflow tests use temporary studies and fabricated QA records separate from participant data; they cover partial saves, reload, stale revisions, completion locking, all-unsure handling, ties and restricted endpoints.
