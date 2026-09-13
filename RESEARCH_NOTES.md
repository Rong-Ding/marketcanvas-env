# Research notes: what can be studied before training?

The object of the first study is the evaluator, not an LLM. A scripted intervention is an experiment when it manipulates a defined property and tests a stated consequence. An LLM is unnecessary for asking whether hiding a headline changes the score. An LLM becomes necessary when asking whether an agent can recognize and repair that defect.

## 1. Operationalize success, then test the measurement

“Good marketing design” is a broad construct. The implementation measures a narrower one: compliance with explicit content, placement, visibility, and contrast requirements. It deliberately does not claim the heuristic is a validated measure of aesthetic preference or conversion potential.

[Amodei et al. (2016), Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) discusses reward hacking as a consequence of an objective that can be satisfied in unintended ways. The practical adaptation here is to construct favorable-looking scores for undesirable canvases: covered text, wrong content with the right role, duplicate roles, and unrelated additions. We do not infer that passing finite examples makes the reward unhackable.

One subtlety: a duplicate CTA can also obscure the earlier label. That intervention tests the combined undesirable outcome, not the isolated causal effect of uniqueness. To isolate uniqueness later, place a second CTA in a separate unoccupied region and match all other geometry. The primary report should not attribute every score change to exactly one internal component.

## 2. Directional and invariance tests

[Ribeiro et al. (2020), CheckList](https://aclanthology.org/2020.acl-main.442/) proposes behavioral testing beyond aggregate accuracy, including minimum functionality, invariance, and directional expectations. We adapt the method to a scalar evaluator:

- Minimum functionality: a known valid reference succeeds; an empty scene fails.
- Directional expectation: lowering required text contrast or misaligning the CTA reduces reward.
- Invariance: changing an unrelated image-placeholder fill preserves reward when it does not affect text backgrounds.

The automated suite includes 12 fixture variants per condition, but these are designed cases, not independent random samples of all marketing designs. Repeated deterministic execution tests reproducibility; it does not increase statistical evidence about a population. Raw effects and exceptions are more useful here than a p-value.

The weak presence-only ablation asks whether checking rendered content adds discrimination on these cases. It is intentionally weak, so the result does not establish superiority over serious alternative evaluators. A more informative extension would compare geometry-only, rendered-visibility, and contrast-aware components on independently designed cases.

## 3. Thresholds, feedback, and reward shaping

The implementation keeps terminal rewards separate from score previews. Repeatedly inspecting a good canvas cannot accumulate reward. The validity cap makes many essential failures tie at -0.02, which may impede learning despite giving the intended ordering relative to a valid design. The contrast sweep makes this trade-off visible.

[Ng, Harada, and Russell (1999), Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) studies when adding reward shaping preserves optimal policies. Potential-based shaping uses `F(s,a,s') = gamma * Phi(s') - Phi(s)`, with appropriate terminal treatment. Arbitrary progress bonuses do not receive that guarantee. We do not implement shaping or claim learning benefits. A later comparison could test sparse terminal reward against a correctly specified potential-based variant while keeping the task objective and evaluation score fixed.

Diagnostic feedback is a separate experimental variable: explaining a contrast failure can help an agent repair it without any parameter update. Comparisons must state what each agent can observe and whether inspection calls consume a separate interaction budget. The core bounds edits; a future agent runner must also bound repeated read-only calls and model tokens.

## 4. Contrast and visibility are different measurements

[W3C's contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html) specifies ratios based on relative luminance. This prototype applies 4.5:1 uniformly rather than implementing the qualifying large-text exception. It also checks geometry and occlusion: perfect foreground/background contrast is insufficient if an opaque image covers the headline.

Visibility uses glyph masks from the same renderer. Foreground antialiasing is excluded from nominal contrast calculation by measuring the underlying background before painting the current glyph. Previously painted text may contribute to that background. Opaque layer coverage is used conservatively for occlusion. General backgrounds, transparency, rich typography, and accessibility beyond required text would need a broader evaluation design.

## 5. What the semantic interface does and does not test

The [Gymnasium API](https://gymnasium.farama.org/api/env/) standardizes lifecycle rather than deciding what information an agent should receive. Complete JSON gives object identity and geometry directly. RGB-only observation would require visual localization and potentially memory. Semantic action success therefore does not establish direct computer-use skill.

The [official MCP SDK v1 documentation](https://py.sdk.modelcontextprotocol.io/v1/) supports the pinned maintenance-line SDK used here. Protocol integration is verified independently through real subprocess tool calls. Protocol correctness is not evidence of agent intelligence. Future throughput workers should bypass interactive protocol overhead while reusing the same simulator.

## Proposed next pilot

1. Freeze this evaluator and collect new canvas pairs from independent authors.
2. Have blinded reviewers score task correctness, legibility, and aesthetic preference separately; adjudicate disagreements rather than hiding them in an average.
3. Inspect disagreements between rubric judgments and heuristic score. Reserve a fresh set before changing weights.
4. Only then run a fixed LLM on matched repair tasks. Record model/version, prompt, observations, feedback condition, token/action budgets, success, latency, and cost.

This first local version completes evaluator development experiments and integration checks. Human validation, LLM comparisons, training, and distributed rollouts remain future work.

## 6. Implemented reward ablations: test the rule, then validate the construct

The [graded-score study](results/reward-comparison/REPORT.md) removes only the essential-validity cap. For 142 below-threshold grayscale contrasts, all 141 adjacent increases are distinguished by the uncapped reward; none are distinguished by the original. However, 72/84 essential-failure fixtures score positively when uncapped. A nonoverlapping second CTA even scores 1.0 because uniqueness was enforced solely through the cap. The alternative is graded in contrast and alignment, not globally continuous: font size, exact content and several layout checks still have thresholds. This is an ablation revealing a trade-off, not a recommended replacement established by training results.

The [aesthetic-rule pilot](results/aesthetic-comparison/REPORT.md) changes a different factor. It retains the cap and adds `H = min(1, headline_size / (target_ratio * CTA_size))`, with default target 1.5 and weight .15. This is a hypothesis for headline-led campaigns. It distinguishes equal-sized versus more prominent headline text among otherwise valid scenes, but also rewards shrinking CTA text to 16px and cannot notice distracting decoration. The study supplies score-hidden visual comparisons and a blank rating sheet, plus 72 paired scenes and nine parameter settings. No human responses are fabricated or inferred from the metric itself.

Adding aesthetic rules is worth testing. [Lok, Feiner and Ngai (2004)](https://www.cs.columbia.edu/~lok/papers/balance.pdf) show how a visual-balance concept can be made computational. [Yang et al. (2024)](https://arxiv.org/abs/2403.18183) study document manipulations including font-size contrast in relation to model confidence. These motivate measurable hypotheses; neither supplies our exact typography ratio or establishes human marketing preference. [ImageReward (Xu et al., 2023)](https://arxiv.org/abs/2304.05977) provides a complementary example of learning visual preferences from human comparisons. It concerns text-to-image generation and is not a validated plug-in evaluator for our canvas.

There are three different claims to separate:

1. **Implementation:** increasing the feature changes the score as specified. Controlled fixtures can establish this.
2. **Perceptual usefulness:** people find the resulting layout clearer or more attractive. Independent preference/readability judgments can test this; judging by the same rule would be circular.
3. **Marketing effectiveness:** the design improves an actual audience outcome. This requires outcome data, such as a suitably controlled campaign experiment; aesthetic preference alone is insufficient.

Logic can prove blind spots: if two scenes have the same measured features, a deterministic function of those features cannot rank them differently. It cannot, by itself, prove that a new rule improves or worsens people's experience. A practical next study freezes the candidates, collects new layouts, randomizes presentation order and side, and compares pairwise reward rankings with human judgments of prominence, CTA readability and overall preference separately. Include ties, reviewer disagreement, and CTA-led briefs. Keep weight-tuning cases separate from final evaluation. Our own exploratory judgments can guide development but should be labeled as such.

## 7. Where scores enter prospective LLM training

The LLM is the policy: conditioned on a task and observation, it generates action/tool-call tokens. The simulator applies them and returns observations. At submission, the chosen evaluator turns the final scene into a scalar. A separate training system uses that scalar to construct an optimization loss. The score need not be differentiable through pixels or simulator code: policy-gradient learning differentiates action log-probabilities, using rewards to estimate which sampled behavior was advantageous. See [Schulman et al. (2017), PPO](https://arxiv.org/abs/1707.06347).

For **PPO**, collect fresh rollouts from a known policy version, retain the exact model context, generated token IDs and action masks, sampling settings, old log-probabilities and rewards, and fit/use value estimates to compute advantages. The learner updates model weights with the PPO objective and any explicitly chosen reference-policy regularization. Tool responses are context, not policy-generated target tokens. Reset independent evaluation tasks to assess the updated policy against a fixed success definition. Our JSON state/action logs support replay, but do not contain those model-specific training fields and are not a ready-made PPO dataset. A terminal-only reward is sufficient in principle; credit assignment over long sequences may be difficult.

For **DPO**, sample alternative continuations from the same task and initial context, score their completed designs, and construct chosen/rejected pairs. Fine-tune relative probabilities of the policy-generated continuations, accounting for tool-turn masking. This is an adaptation of preference learning to interactive trajectories, not automatically supplied by standard single-response DPO code. Score-derived labels are synthetic preferences and inherit the heuristic's errors. Standard DPO does not require a separately learned reward model or an online PPO loop. See [Rafailov et al. (2023)](https://arxiv.org/abs/2305.18290).

A **ReST-style** workflow samples trajectories, filters or weights them using the evaluator, and improves the policy from selected generated data before sampling again. See [Gulcehre et al. (2023)](https://arxiv.org/abs/2308.08998). Human demonstrations can also supply supervised action examples; a high-scoring final scene does not guarantee every preceding action is worth imitating. [Stiennon et al. (2020)](https://arxiv.org/abs/2009.01325) offers an accessible language-task precedent for the broader pipeline: human comparisons, reward modeling, and RL fine-tuning. Here our heuristic initially takes the reward model's scoring role.

Giving a frozen LLM a score or failure explanation and asking it to revise is **inference-time feedback**, with no parameter update. Selecting the best of several scored designs is also useful without training. These experiments should log which diagnostics the agent sees. Changing a terminal score and exposing additional feedback are separate interventions.

Finally, a graded final score and rewards paid at each step are different design choices. Accumulated improvement can be reflected in the final scene score. Naively paying only positive score deltas permits damage/repair cycles. Potential-based shaping instead uses `r_shaped = r_base + gamma * Phi(next) - Phi(current)`, with zero terminal potential and a matching discount in this episodic setting. For a fixed initial state its discounted shaping contribution is a policy-independent offset, under the standard assumptions; our uncapped terminal replacement has no such guarantee. Shaped step rewards also need not remain in [-1,1]. We preserve the assignment's original terminal reward and leave shaping/training as future extensions.

## 8. Implemented viewer pilot (completed; historical design notes)

The proposed early-viewing study is now implemented as a separate, frozen pilot with twelve matched pairs. The only planned human reviewer participated in formulating the hypothesis. The local page on port 8766 collects perceived CTA discoverability and overall preference separately, saves responses outside the study pack, and withholds predictions until explicit completion. It does not measure gaze or search time, and no human outcome is claimed in advance.

The model uses rendered-region area and edge contrast, physical distance, and a revisit penalty. Exact probability propagation estimates whether the CTA has been inspected within one, two or three model glances. Semantic roles are excluded from selection; simulator segmentation is still privileged information. The reward extension uses the two-glance probability and retains essential validity checks. Its parameters are engineering hypotheses, not fitted psychophysical constants. The complete specification is in [VIEWER_METHOD.md](docs/VIEWER_METHOD.md), with a [short report draft](docs/VIEWER_PILOT_WRITEUP.md).

The study pack freezes planned stimuli, predictions, sensitivity settings, source and documentation before human ratings. The automated analysis reports ties, unsure responses, directional coverage and pair-level disagreements. Test answers are isolated from the human response file. Any changes prompted by ratings must be reported as exploratory and evaluated on new material.


## 9. Completed results and submission route

The current account is [the companion paper](docs/RESEARCH_PAPER.md). It includes the initial pilot, the 12-pair batch with six embedded consistency checks, and exploratory reward fitting. The viewer and fitted rewards did not establish improved human alignment. Reproduction uses the minimal released annotations described in [research/README.md](research/README.md); private notes and account data are excluded. Frozen source documents retain their original pre-annotation wording for provenance.
