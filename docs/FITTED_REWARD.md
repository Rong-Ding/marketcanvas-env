# Exploratory fitting of a local-context reward

We reuse the first presentations of 12 original and 12 embedded pairs from one
reviewer. The six embedded repeat responses diagnose consistency and are excluded
from training and validation. Only CTA discoverability is fitted; overall design
preference remains a separate construct. The reviewer and analyst have already
seen these results. This is post-data model development, not a preregistered test.

## Features and identifiability

Each canvas produces four proposed features: CTA width / canvas width; absolute
linear-light relative-luminance difference between the CTA fill and its nearest
filled neighbor above; Euclidean center-to-center distance to that neighbor / canvas diagonal; and
maximum font size of non-headline text / canvas height. Choose the nearest
filled element wholly above the CTA by center distance; horizontal overlap is
not required. This implements the reviewer’s clarification of object-to-object
center distance, not distance from a simulated gaze location. These are simple simulator geometry/style proxies, not a validated
perceptual model. The luminance feature ignores hue differences; competing text
size is a coarse addition to cover the existing text-size manipulation.

Only within-pair differences identify weights from these ratings. A feature
with no within-pair variation is excluded and its coefficient fixed to zero,
explicitly marked unidentified. Center distance varies in the image-position comparisons. A zero excluded
coefficient would indicate missing evidence, not proof that a feature is irrelevant.
Color difference also co-varies with image/background contrast, so the fit cannot
separate contextual color separation from competing-image salience. Width also
changes button area. Do not interpret coefficients as isolated causal effects.

## Model and objective

Canvas utility is u(s) = w dot (f(s) - c), with c the training-feature mean.
For a left/right pair, d = u(left) - u(right). The three response probabilities
are softmax(d/2, -d/2, b) for left, right, and tie. This symmetric, Davidson-form
extension of pairwise preference learning uses a learned tie logit b and no
screen-side intercept. A higher score means predicted CTA discoverability for
this reviewer, not universal preference or marketing effectiveness.

Fit mean negative log likelihood plus 0.1/2 times the sum of squared standardized
weights and squared b. Standardization uses training-only RMS feature differences;
no hyperparameter search is performed. Deterministic Newton updates with line
search converge to the convex penalized objective. The hard comparison chooses
the largest probability; if maxima tie exactly, return tie rather than inventing
a side preference. In particular, indistinguishable feature vectors can yield
equal directional probabilities: this decision convention is not proof of a
human perceptual tie. Report likelihood as well as hard agreement.

## Reward interface

An optional bounded component is Qfit(s)=sigmoid(u(s)); this is a score mapping,
not a measured probability that a human will notice the button. Combine it as
R(s)=2[(1-alpha)Qoriginal(s)+alpha Qfit(s)]-1, with alpha=0.15 fixed, not fitted.
Keep the original essential-constraint cap Q<=0.49 for invalid designs. Missing
required feature geometry falls back to the original reward with supported=false.
Values outside training feature ranges are flagged. Neither marginal range checks
nor the validity cap ensure safe generalization or prevent all reward hacking.

Pair choices identify score differences, not an absolute quality origin or
the right mixture weight. Centering, sigmoid scale and alpha are implementation
conventions; they should not be presented as human-calibrated cardinal rewards.
The new evaluator is opt-in and does not change the served canvas or RL default.
Oversized buttons, distracting text and extreme colors remain optimization risks.

## Evaluation and future training

Report full-data fit separately from three-fold grouped cross-validation: hold
out fixture seed 0, 1, or 2 across both batches and all manipulation families,
refitting feature scales, weights and b inside each fold. Each fold holds out
eight first presentations. Related templates remain across folds; feature design
was informed by the full dataset. Thus this is a diagnostic of fitting stability,
not untouched validation, participant generalization, or a significance test.

Freeze the fitted model before genuinely new evaluation. A future LLM rollout
could receive this scalar at episode end through an explicitly chosen evaluator;
PPO or another trainer would then update the policy separately. This experiment
trains only a tiny reward model from annotations, not an LLM and not a gaze model.

## Sources and relation to prior work

Pairwise reward differences are the basis of Bradley–Terry preference learning:
[Stanford, Learning from Preferences](https://web.stanford.edu/~jurafsky/slp3/9.pdf).
Ties deserve explicit treatment; [Liu, Ge & Zhu, Reward Learning From Preference
With Ties (2024)](https://arxiv.org/abs/2410.05328) studies a different tie extension
(Rao–Kupper). We use that paper as motivation, not a claim to reproduce its method
or results. Group separation and training-only preprocessing follow standard
[cross-validation guidance](https://scikit-learn.org/1.5/modules/cross_validation.html).
