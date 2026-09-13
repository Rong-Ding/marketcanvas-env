"""Official MCP SDK adapter. One isolated environment per stdio server process.

No access to files, network services, company systems, or model credentials is exposed.
"""
import threading
from typing import Any
from mcp.server.fastmcp import FastMCP
from .env import MarketCanvasEnv
from .models import ACTION_ADAPTER

mcp = FastMCP("MarketCanvas", instructions="Inspect the task and canvas, edit via execute_action, then submit. Reward previews are diagnostics, not earned rewards. Reset starts a new episode.")
env = MarketCanvasEnv()
lock = threading.RLock()


@mcp.tool()
def get_canvas_state() -> dict[str, Any]:
    """Read complete semantic state, task constraints, geometry, and spatial relations."""
    with lock:
        return env.observe()


@mcp.tool()
def get_action_schema() -> dict[str, Any]:
    """Read the validated action schema for add, move, update, delete, and submit."""
    return ACTION_ADAPTER.json_schema()


@mcp.tool()
def execute_action(action: dict) -> dict[str, Any]:
    """Execute one edit or submit; an invalid attempt consumes one step without changing elements.

    Examples: {"op":"move_element","id":"e1","x":100,"y":80};
    {"op":"update_element","id":"e1","properties":{"text_color":"#000000"}};
    {"op":"add_element","element":{"type":"text","role":"headline","content":"Summer Sale"}};
    {"op":"submit"}. Call get_action_schema for full property definitions.
    """
    with lock:
        obs, reward, terminated, truncated, info = env.step(action)
        return {"observation": obs, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}


@mcp.tool()
def get_current_reward() -> dict[str, Any]:
    """Read a diagnostic score preview and components. Does not advance or reward an episode."""
    with lock:
        return {"kind": "preview" if not env.state.terminated else "final_evaluation", **env.current_reward()}


@mcp.tool()
def reset_environment(seed: int = 0, task: dict | None = None) -> dict[str, Any]:
    """Discard this episode and start a blank canvas with an optional structured task specification."""
    with lock:
        obs, info = env.reset(seed=seed, options={"task": task or {}})
        return {"observation": obs, "info": info}


if __name__ == "__main__":
    mcp.run(transport="stdio")
