"""Verify an exported demo JSONL or inspector JSON trajectory by exact replay."""
import argparse
import json
from pathlib import Path
from marketcanvas import MarketCanvasEnv


def replay(path):
    text=Path(path).read_text()
    # Try whole-file JSON first, then the original JSONL demo format.
    try:
        data=json.loads(text)
    except json.JSONDecodeError:
        data=[json.loads(line) for line in text.splitlines() if line.strip()]
    versioned=isinstance(data,dict) and data.get("format_version")==2
    if versioned:
        records=data["transitions"]
        initial=data["initial_observation"]
    else:
        records=data if isinstance(data,list) else [data]
        if not records:
            raise ValueError("Legacy trajectory is empty and has no initial scene")
        initial=records[0]["observation"]
    env=MarketCanvasEnv(task=initial["task"],max_steps=initial["max_steps"])
    env.reset(seed=initial["seed"])
    if versioned:
        for action in data["setup_actions"]:
            if env.step(action)[4]["error"]:
                raise ValueError("Invalid setup action in trajectory")
        env.begin_prepared_episode()
        if env.observe()!=initial:
            raise ValueError("Initial scene does not match recorded setup")
    for index,record in enumerate(records):
        if env.observe()!=record["observation"]:
            raise ValueError(f"Observation mismatch at step {index}")
        obs,reward,term,trunc,info=env.step(record["action"])
        expected=(record["next_observation"],record["reward"],record["terminated"],record["truncated"],record["info"])
        if (obs,reward,term,trunc,info)!=expected:
            raise ValueError(f"Transition mismatch at step {index}")
    return env


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("trajectory")
    args=parser.parse_args()
    env=replay(args.trajectory)
    print(f"Replay verified: {env.state.steps} actions; state SHA256 {env.state_hash()}")
