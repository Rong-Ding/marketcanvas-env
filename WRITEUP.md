# MarketCanvas-Env: answers to the assignment questions

## 1. Why did you choose these Action and State spaces?

I chose high-level semantic actions: add, move, update, delete and submit. They let us evaluate whether an agent can turn a design brief into a valid layout without also solving mouse targeting, selection and keyboard focus. They are straightforward to validate and expose through MCP. The trade-off is reduced realism: success here does not demonstrate that an agent can operate Canva through screenshots and mouse clicks.

The semantic state describes the 800 x 600 canvas, every element's properties, spatial relationships, task constraints, stable identifiers, remaining action budget and episode status. This makes the environment fully observable. The remaining budget matters because the same design can require different decisions with one action left versus twenty. An optional RGB image renders the same scene for future multimodal experiments. A screenshot-only agent would receive less information and need visual grounding.

The Python API, Gymnasium adapter, MCP server and inspector share one deterministic simulator. Tasks use explicit headline, CTA, color and alignment constraints, rather than arbitrary prompt understanding. Invalid edits consume an action but preserve the scene. Submission or the intrinsic 40-action budget ends the episode [1]. Prepared examples record setup separately; construction episodes start blank. The Gymnasium adapter declares JSON-text spaces, not a ready-made PPO policy interface.

## 2. How does your reward work, and what are its potential loopholes?

The environment calculates quality q = 0.45C + 0.20V + 0.20A + 0.15L and returns terminal reward R = 2q - 1. Each component is between zero and one:

**C - constraints:** the required headline, CTA text and CTA fill are correct.

**V - visibility:** required text fits, remains at least 99% visible, and uses at least 16px text.

**A - accessibility:** foreground/background contrast approaches the required ratio. The prototype applies 4.5:1 to all required text using linearized sRGB; this is not full WCAG conformance [2].

**L - layout:** the headline and CTA follow the requested alignment and spacing.

If essential content, readability, contrast or role uniqueness fails, q is capped at 0.49. This prevents an invalid design from earning positive reward by doing well on easier criteria. Intermediate actions return zero; termination delivers the reward once. The reward preview is diagnostic feedback, not additional earned reward.

The main loophole is that meeting these rules is not equivalent to good marketing design. An agent can add irrelevant decoration, create an unbalanced composition or exploit exact thresholds while retaining a high score. The validity cap also hides incremental improvements below its threshold. Controlled perturbations tested 132 scenes: all 120 specified directional/invariance expectations passed, while 12 unrelated-text probes exposed an unresolved loophole.

**Human-evaluation pilot - Motivation.** Constraint satisfaction does not establish perceptual quality. Research on viewing sequences [3] and aesthetic preference [4] motivated testing an attention-based reward against perceived CTA discoverability and overall preference as separate outcomes.

**Methods.** One researcher-reviewer assessed 12 new pairs varying button width, image contrast, image position and competing-text size. Six interleaved repeats assessed consistency: three same-side and three reversed-side. First-presentation orientation was balanced; scores and repeat identities were hidden. Reward agreement excluded repeats.

**Results.** Viewer-reward agreement was 4/12 for each outcome. Same-side consistency was 3/3 for discoverability and 2/3 for preference; reversed-side consistency was 1/3 for both. A prespecified variant without gaze-distance cost matched 7/12 discoverability judgments by matching three image-position ties.

**Interim interpretation.** These findings identify distance weighting and response stability as targets for investigation, without validating the reward. The reviewer helped develop the hypotheses and had seen initial-pilot results; three repeats per type cannot establish a causal position effect. Increased sensitivity alone does not establish human alignment. The companion paper details the unsuccessful reward fit.

<!-- pagebreak -->

## 3. What would bottleneck PPO with 10,000 parallel VLM rollouts, and how would you redesign the environment?

I would expect two major pressures: generating actions with the model and handling many visual observations. Model weights and retained conversation context consume accelerator memory; rendering, copying and transferring images add cost. An uncompressed 800 x 600 RGB image occupies 1.44 MB. Keeping one image per environment requires 14.4 GB; retaining thirty images per environment requires 432 GB, excluding model memory. These are raw storage estimates, not measured distributed performance.

**First, reduce unnecessary image generation and storage.** If a canvas has not changed, reuse its rendered image. Because the simulator is deterministic, action logs with occasional complete snapshots can replace some duplicated histories: earlier observations can be reconstructed by replay. This trades storage for computation, and a trainer may still need some observations readily available. Rendering-based reward checks also need profiling, even when the agent receives semantic observations.

**Second, let fewer worker programs manage multiple canvases.** Rather than launch 10,000 separate servers, each worker can hold several independent canvas states and apply actions to the appropriate one. This reduces the overhead of managing separate programs while keeping episodes separate. Workers can call the Python simulator directly; MCP remains available for external clients. Profiling would determine the number of workers and canvases per worker.

**Third, process several ready model requests together.** When multiple canvases ask for their next action, the inference system can group their requests so the GPU makes progress on them together. Each retains its own input and answer; batching shares computation, not conversation context. When one request finishes, another waiting request can take its place without waiting for all episodes to finish. Limit the waiting queue so workers cannot generate requests faster than inference can handle them indefinitely. The aim is more completed actions per second, not necessarily a faster response for every canvas.

**Finally, retain the information PPO needs.** The training system should record which policy version generated each action and its probability under that policy. It must distinguish model-generated action tokens from tool responses and control how old collected trajectories can become as the policy changes. These additions belong to the training system; the environment alone does not train an LLM. Deterministic resets, action ordering and worker recovery must survive the redesign. No distributed throughput or model-training improvement is claimed.

## References

1. [Farama: Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
2. [W3C: Contrast (Minimum)](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).
3. [Chakraborty et al.: Predicting Visual Attention in Graphic Design Documents](https://arxiv.org/abs/2407.02439).
4. [Van Geert, Ding & Wagemans (online 2024; issue 2025): A Cross-Cultural Comparison of Aesthetic Preferences for Neatly Organized Compositions](https://doi.org/10.1177/02762374241245917).
