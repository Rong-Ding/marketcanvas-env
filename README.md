# MarketCanvas-Env

A standalone Python design-canvas simulator with a local inspector, an RL-style API, an optional Gymnasium adapter, and an official-SDK MCP server. No model, API key, external data, or training run is required.

## Start here

Read [WRITEUP.md](WRITEUP.md) or its [two-page PDF](output/pdf/WRITEUP.pdf) for the assignment response. The optional [companion research paper](docs/RESEARCH_PAPER.md) reports the human study with embedded consistency checks and exploratory reward fitting.

For a new checkout, follow **Fresh installation** below before running the demo or inspector.

## Try it locally

After installation, run from this directory:

```sh
.venv/bin/python playground.py
```

Open **http://127.0.0.1:8765/**. The inspector initially shows a valid reference banner. Try:

1. Choose **White text on yellow CTA** and click **Load example**. Reward should fall from `1.000` to `-0.020`.
2. Select element **e3 · cta · shape**, set **Text color** to `#172033`, and click **Apply edit**. Reward returns to `1.000`.
3. Click **Submit design**. The episode ends and delivers its terminal reward once.
4. Try **Covered headline**, **Clipped headline**, **Duplicate CTA**, or **Unrelated extra text (known limitation)**.
5. Use **Start blank** to design manually. Add elements and assign a role with the role selector. Required headline type is `text`; CTA type is `shape`.

Fill and text colors each offer named presets, a color preview, and an editable six-digit hex code. For example, `#000000` is black, `#FFFFFF` is white, and `#FFFF00` is yellow. Choose **Apply edit** to commit a color change. Text can use **No fill**; shapes and image placeholders require a fill.

The controls invoke the actual Python environment. The canvas image is rendered by Pillow, not a separate browser approximation. The inspector and stdio MCP server use the same code but own separate episodes. Stop the inspector with Ctrl+C. To use a different port: `python playground.py --port 8766`.

Loaded examples start at **0/40**, including examples with a prepared defect. Their setup actions are recorded separately and do not consume your edit budget. Every edit attempt and submission then consumes one action. Inspector trajectory downloads contain `initial_observation`, `setup_actions`, and `transitions` in a versioned JSON object, so even an untouched example can be replayed. `replay.py` also accepts the original JSON-array and JSONL exports. The scripted construction demo still counts its own construction actions because it deliberately builds from blank.

## Fresh installation

Use Python 3.12 or later; this version was tested on Python 3.12.14, macOS arm64. From the project directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

Alternatively, use `uv venv --python 3.12` then `uv pip install -r requirements.lock.txt`.
A fresh environment and clean source copy passed 74 tests (one private-data integration check skipped), the demo, replay, and public-data research reproduction. See [verification details](docs/VERIFICATION.md). The lock records the exact tested package versions; other operating systems have not been tested. The bundled DejaVu font has its original license in `marketcanvas/assets/`.

## Demo, experiments, and tests

```sh
.venv/bin/python demo.py
.venv/bin/python demo.py --headline "Weekend Offers" --cta "Explore offers"
.venv/bin/python replay.py results/demo/trajectory.jsonl
.venv/bin/python experiments.py --seeds 12
.venv/bin/python reward_comparison.py --seeds 12
.venv/bin/python aesthetic_comparison.py
.venv/bin/python -m pytest -q
```

`demo.py` generates its mock prompt from explicit constraints, builds a banner, introduces poor contrast, repairs it, submits, and prints state and reward. Outputs include before/after PNGs, final state and evaluation, and a JSONL trajectory.

The experiment report is [results/experiments/REPORT.md](results/experiments/REPORT.md), with raw CSVs, per-condition JSON and PNGs, a visual comparison sheet, and measured local timings. These are tests of the evaluator, not LLM performance results. Re-running overwrites generated files in the chosen output directory; use `--output` to retain another run.

Two separate, offline reward studies preserve the original live reward:

- [Graded-reward comparison](results/reward-comparison/REPORT.md): remove only the validity cap, measure contrast-repair sensitivity, and expose invalid high-scoring designs. Includes a replayable repair trajectory and separately logged alternative scores.
- [Aesthetic-rule pilot](results/aesthetic-comparison/REPORT.md): add a headline-size hierarchy rule while retaining the cap, inspect six score-hidden designs, and test counterexamples and parameter sensitivity. Includes a blank review form; no human ratings are claimed.

The alternative evaluators are `evaluate_uncapped(state)` and `evaluate_hierarchy(state)` in `marketcanvas/reward_variants.py`. They do not mutate episodes or change the reward delivered by the playground, MCP, or Gymnasium adapter. A future trainer must explicitly select and version its terminal evaluator. Neither is a globally continuous or validated measure of marketing effectiveness.

## Viewer research pilot

```sh
.venv/bin/python viewer_pilot.py
.venv/bin/python rating_server.py
```

Open **http://127.0.0.1:8766/** for twelve score-hidden design pairs. Rate CTA discoverability and overall preference separately. Choices save locally; notes save on leaving the field or using Save progress. You can resume after reloading. Finishing explicitly reveals the predictions and locks ratings. The canvas inspector on port 8765 retains its own scene and original reward.

The builder freezes stimuli, parameters, predictions, relevant source and documentation in `results/viewer-pilot-v1/`. Re-running verifies this pack rather than replacing it. New hypotheses require a new study directory and separate responses. Personal ratings live in `private/viewer-pilot-v1/ratings.json`, excluded from Git. Completed analysis is generated in the adjacent `analysis/` folder and is also shown in the rating page. No human results are claimed before completion.

Read the completed [research paper](docs/RESEARCH_PAPER.md) and [full viewer method](docs/VIEWER_METHOD.md). Reproduce the reported results with `python reproduce_research.py`; see [research/README.md](research/README.md). The principal follow-up uses `embedded_pilot.py --serve` on port 8768: 12 new pairs plus six embedded checks, with repeats excluded from primary agreement. The pure Python alternative is `marketcanvas.viewer.evaluate_viewer(state)`. The model predicts early region inspection, not actual gaze, comprehension, or marketing effectiveness. It is not trained on human data. Automated workflow tests use separate temporary records, never the participant's ratings.

## Research results and scope

The embedded batch produced viewer agreement of 4/12 for discoverability and 4/12 for preference. Same-side discoverability consistency was 3/3; reversed-side consistency was 1/3. These are exploratory results from one reviewer, not a validated viewer or a demonstration of side bias. The later fitted contextual reward achieved 10/24 training agreement and 8/24 grouped cross-validation agreement, versus 11/24 for the frozen viewer. It remains opt-in in `marketcanvas/fitted_reward.py`; the live reward is unchanged. See [research/README.md](research/README.md) for provenance and reproducible outputs.

## RL interface

```python
from marketcanvas import MarketCanvasEnv

env = MarketCanvasEnv()
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step({
    "op": "add_element",
    "element": {"type": "text", "role": "headline", "content": "Summer Sale",
                "x": 100, "y": 80, "width": 600, "height": 90, "font_size": 42}
})
preview = env.current_reward()  # Read-only diagnostic; not an earned reward.
rgb = env.render()             # uint8 array, shape (600, 800, 3).
```

Actions: `add_element`, `move_element`, `update_element`, `delete_element`, `submit`. Updating properties includes text, colors, size, role, type, and z-index. IDs/creation order are immutable. Every attempted edit costs one step; invalid actions preserve elements and return an error. Post-terminal actions raise `EpisodeFinished`. There is no auto-reset.

The 40-action limit is part of the task and remaining time is observed. `submit` or budget exhaustion sets `terminated=True`; the core never independently truncates. Nonterminal reward is zero. Terminal reward lies in [-1,1] and is returned once. Reset begins a new trial, not an action available inside a training episode.

`GymCanvasEnv` in `marketcanvas/gym_adapter.py` passes Gymnasium's environment checker and declares JSON-text action/observation spaces. Actions and observations are JSON **strings** in this adapter; the core uses dictionaries. This does not supply a tokenizer, a tensor policy, or an off-the-shelf PPO training integration.

## MCP

```sh
.venv/bin/python mcp_server.py
```

This command waits for stdio protocol messages; it is not a browser server. Use `mcp-config.example.json` with an MCP-capable desktop client, replacing both paths with absolute paths. Do not add ordinary stdout logging to this server.

Tools: `get_canvas_state`, `get_action_schema`, `execute_action`, `get_current_reward`, `reset_environment`. First read the state and schema, then execute actions and submit. Tool responses include structured JSON. One stdio process owns one environment; edits are serialized with a lock. The transport test launches actual MCP subprocesses, performs initialization and tool discovery, compares their edits with Python execution, and checks independent processes.

The inspector also feature-detects browser WebMCP and exposes two local page tools. That is an optional convenience; the required standalone MCP server works independently of browser support.

## Scope and limits

800×600 fixed canvas; maximum 24 elements; single-line printable ASCII text, up to 256 characters; fixed DejaVu Sans font; opaque rectangles and image placeholders; no rotation, transparency, rich text, imported images, or mouse interaction. Shapes may contain centered labels. Text is centered within its own bounding box. Bounds are validated; text overflow is rendered clipped and penalized when it affects a required slot.

The agent gets exact semantic state and constraints. This favors controlled task evaluation over realistic screenshot-only computer use. Required text is checked for content, glyph visibility, clipping, minimum 16px size, and contrast. Decorative content is not comprehensively evaluated. A high reward is not a certification of aesthetic quality, marketing effectiveness, or complete accessibility.

Read [WRITEUP.md](WRITEUP.md) for formulation and scaling, and [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for literature, experimental interpretation, and next studies.
