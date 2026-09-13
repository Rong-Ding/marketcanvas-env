# Do more sensitive canvas rewards better match human judgments?

A single-reviewer development study with embedded repeat checks

## Abstract

A design environment needs an evaluator that distinguishes useful agent behavior, but additional sensitivity need not improve validity. We compared a constraint-based reward with an attention-inspired extension using pairwise human judgments of CTA discoverability and design preference. The principal batch contained 12 new pairs and six embedded repetitions. The extension matched 4/12 first-presentation judgments for each outcome. Discoverability consistency was 3/3 for same-side and 1/3 for reversed-side repeats. A prespecified distance-free variant matched 7/12 discoverability judgments, principally through ties. A subsequent contextual reward fitted to 24 first presentations across two batches did not improve agreement. These exploratory results illustrate the value of distinguishing constructs, auditing response consistency and reporting unsuccessful reward refinement.

## Motivation and research question

Graphic-design attention is a sequence of inspections, not only a static saliency map. Chakraborty et al. model saliency and fixation sequences using inverse reinforcement learning [1]. This motivates a sequential proxy but does not validate the simplified model here. Empirical aesthetics also resists reducing overall preference to one visual attribute: Van Geert, Ding and Wagemans found shared and differing relationships involving order, complexity and appreciation across sampled language groups [2]. We therefore measure perceived CTA discoverability separately from overall preference, without calling either a direct measure of gaze, comprehension or marketing conversion.

The question is whether an attention component can distinguish valid canvases in ways that agree with a reviewer, and whether apparent agreement survives basic response-consistency checks. Behavioral perturbations [3] motivate controlled manipulations. The reviewer contributed to the hypotheses; this is development research, not an independent participant study.

## Environment and candidate evaluator

The deterministic 800 x 600 simulator provides text, shapes and image placeholders. Its original quality q0 combines content constraints, required-text visibility, contrast and layout; invalid essential constraints cap quality at 0.49 before conversion to reward 2q - 1. All human-study pairs were designed to satisfy the original rubric and receive reward +1.0, deliberately examining its ceiling among valid scenes.

The viewer derives each element's visible support by removing it and identifying changed pixels. Size is normalized square-root support area; contrast is normalized mean nonzero luminance-edge strength. A region's prominence is their equal-weight mixture. Next-region probability is proportional to exp(2 x prominence - gaze distance - visited indicator). Distance uses physical canvas aspect ratio, normalized by its diagonal; the initial simulated gaze is central. Exact propagation enumerates region/visited-set states for up to three glances. Region selection has no task role or semantic content input; the CTA is identified afterward.

Let D2 be the probability of visiting the CTA by glance two. The alternative quality is 0.85q0 + 0.15D2, retaining the essential-validity cap. This borrows sequential structure without learning from eye movements. Privileged element segmentation, arbitrary parameter choices, simplistic visual features and lack of semantics limit interpretation. The original live environment reward remains unchanged.

<!-- pagebreak -->

## Human annotation and embedded checks

An initial batch used 12 pairs: four manipulations (image contrast, competing-text size, CTA width and image position) crossed with three campaign variants. After results were discussed, a new batch used altered campaigns and geometry with the same manipulation families. It contained 12 first presentations and six embedded repeats. Same-side repeats assessed ordinary response stability; reversed-side repeats assessed stability under changed presentation. The repeated sources covered all four families but were different pairs across repeat types, limiting causal comparison.

The second schedule was fixed before collection: seed 2026091401; six first presentations per condition orientation; three repeats of each type; at least four item positions between a first and repeat presentation. New and repeated items were interleaved under neutral IDs. Stimuli, model predictions and relevant source were frozen with hashes. The reviewer knew checks existed and had seen earlier aggregate results, but received neither current model scores nor repeat labels while responding. Finishing locked ratings before revealing scores. No deception or full double-blind claim is made.

Each item asked (1) which action button is easier to find, and (2) which design is preferred overall, allowing left/right/equal/unsure. Primary agreement used only the 12 first presentations. Equal matched only equal; unsure would be excluded and counted, but none occurred. Repeat consistency compared image identity after reversing answer mapping where appropriate. Repeats were never counted as additional independent design pairs. No gaze, response-time or marketing-outcome data were collected.

## Results

| First-presentation evaluator | Discoverability | Overall preference |
|---|---:|---:|
| Original heuristic (all ties) | 4/12 | 3/12 |
| Primary viewer extension | 4/12 | 4/12 |
| Prespecified no-gaze-distance variant | 7/12 | 4/12 |

The reviewer selected equal on four discoverability and three preference items. Thus baseline agreements reflect ties, not successful ranking. On directional human choices only, the primary viewer matched 4/8 discoverability and 4/9 preference judgments. The distance-free variant gained three discoverability matches by predicting equal for image-position manipulations, as the reviewer did. It did not improve directional discrimination. Most other prespecified parameter variants retained the primary rankings.

| Repeat consistency by image identity | Discoverability | Preference |
|---|---:|---:|
| Same-side (three checks) | 3/3 | 2/3 |
| Reversed-side (three checks) | 1/3 | 1/3 |

Two reversed discoverability checks retained the displayed side, thereby changing the selected design. Reversed preference responses comprised one same-design choice, one same-side choice, and one directional-to-equal change. Aggregate first-presentation choices were balanced for discoverability (4 left, 4 right, 4 equal), demonstrating that aggregate side counts alone need not reveal presentation sensitivity. Three checks of different items cannot distinguish position effects from item difficulty, memory, feedback or ordinary variability.

For context, the initial batch's viewer agreement was 7/12 discoverability and 3/12 preference; the original reward tied all pairs and the reviewer chose no ties. Its 11/12 left discoverability responses helped motivate the added controls. Because feedback and stimuli changed, differences between batches are not treatment effects.

<!-- pagebreak -->

## Exploratory fitting from existing judgments

The reviewer suggested that button extent and its relationship to nearby objects might matter, clarifying distance as object-center separation rather than gaze travel. We subsequently fitted a small contextual model to discoverability labels from 24 unique first presentations across both batches; all six repeat presentations were excluded. Features were normalized CTA width, luminance difference from the nearest filled object above, Euclidean center-to-center distance, and competing non-headline text size. Luminance is only a proxy for color difference; width co-varies with area, and image/CTA contrast co-varies with image/background salience.

Utility is linear in centered features. For utility difference d between two designs, response probabilities are softmax(d/2, -d/2, b) for left/right/equal, with a learned tie parameter b. Explicit treatment of ties is motivated by preference-learning research [4]; our symmetric softmax form is not a replication of that paper's Rao-Kupper method. Mean negative log likelihood plus fixed ridge penalty 0.1 is minimized; exact probability maxima ties return equal without a side preference. No screen-side intercept or aesthetic-preference label is fitted.

The fitted weights favor wider buttons, weakly favor shorter center distance and smaller competing text, and assign zero weight to luminance difference in the full-data fit. Agreement is 10/24 on fitted data, compared with 11/24 for the frozen viewer on those pairs. Three-fold cross-validation groups fixture seed across both batches and all families, refitting scales and parameters on 16 pairs before predicting eight held-out pairs. Agreement is 8/24; log loss is 1.168 versus 1.099 for uniform three-choice probabilities. Related templates remain across folds and the hypothesis was informed by all responses: this is a diagnostic, not independent validation.

The optional reward maps utility through a sigmoid and blends it with original quality at fixed weight 0.15, retaining the validity cap. Its origin, scale and mixture weight are conventions, not identified cardinal human rewards. This trains a tiny reward model, not an LLM. Coefficients and predictions are retained without promoting the model to the environment default.

## Interpretation and implications for agent training

This investigation does not validate the viewer extension or fitted reward. Its contribution is methodological: distinguish validity from perceived discoverability and preference; test what new reward sensitivity measures; and examine annotation stability before attributing disagreement wholly to a model. The human task names a CTA, whereas the sequential proxy lacks semantic search. Reported preference is also not equivalent to conversion performance.

The no-distance result provides a targeted hypothesis about our model's assumptions, not evidence that distance is irrelevant to human vision. A future study should independently vary center separation, visual contrast and competing salience, evaluate indifference explicitly, and recruit reviewers uninvolved in hypothesis construction. New held-out material is needed after fitting. Present repetitions must not be averaged into extra independent evidence or selectively discarded.

For future PPO-style training, the chosen evaluator would supply terminal scalar feedback while a separate trainer updates the LLM policy. Because optimization can exploit a misspecified score, measured agreement and consistency are part of environment design rather than decorative evaluation. No agent-performance improvement, production scalability or population-level significance is claimed.

## References and reproducibility

1. [Chakraborty et al., Predicting Visual Attention in Graphic Design Documents](https://arxiv.org/abs/2407.02439). Journal DOI: 10.1109/TMM.2022.3176942; arXiv deposit 2024.
2. [Van Geert, Ding & Wagemans, A Cross-Cultural Comparison of Aesthetic Preferences for Neatly Organized Compositions](https://doi.org/10.1177/02762374241245917). Online 2024; journal issue 2025.
3. [Ribeiro et al. (2020), Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/). Adapted as a testing approach, not direct evidence about visual design.
4. [Liu, Ge & Zhu (2024), Reward Learning From Preference With Ties](https://arxiv.org/abs/2410.05328).

Frozen study packs, fixed schedules and executable evaluators accompany the code. See research/README.md for the minimal annotation release and reproduction command. No personal notes or account information are included. Lower-level implementation details are in VIEWER_METHOD.md, EMBEDDED_PILOT.md and FITTED_REWARD.md.
