"""Run: python demo.py [--headline 'Weekend Offers'] [--output results/demo]."""
import argparse
import json
from pathlib import Path
from marketcanvas.models import TaskSpec
from marketcanvas.scenarios import build_reference, perturb, apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline", default="Summer Sale")
    parser.add_argument("--cta", default="Shop now")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output", default="results/demo")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    task = TaskSpec(headline=args.headline, cta=args.cta)
    print("BRIEF:", task.prompt())
    env = build_reference(args.seed, task)
    perturb(env, "low_contrast")
    before = env.current_reward()
    env.save_png(out / "before.png")
    print("Before repair:", json.dumps(before, indent=2))
    apply(env, {"op": "update_element", "id": "e3", "properties": {"text_color": "#172033"}})
    obs, reward, terminated, truncated, info = apply(env, {"op": "submit"})
    env.save_png(out / "after.png")
    (out / "state.json").write_text(json.dumps(obs, indent=2)+"\n")
    (out / "trajectory.jsonl").write_text("".join(json.dumps(t)+"\n" for t in env.trajectory))
    (out / "evaluation.json").write_text(json.dumps(info["evaluation"], indent=2)+"\n")
    print("FINAL STATE:", json.dumps(obs, indent=2))
    print(f"Final reward: {reward}; terminated={terminated}; truncated={truncated}")
    print("Saved before/after images, state, evaluation, and replayable trajectory to", out.resolve())


if __name__ == "__main__":
    main()
