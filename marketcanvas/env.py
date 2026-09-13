"""RL-style lifecycle. Inspection is read-only; rewards are paid once at termination."""
import copy
import hashlib
import json
from pydantic import ValidationError
from .models import ACTION_ADAPTER, Element, ElementSpec, State, TaskSpec, observation
from .rewards import evaluate
from .rendering import render_scene


class EpisodeFinished(RuntimeError):
    pass


class MarketCanvasEnv:
    def __init__(self, task=None, max_steps=40):
        if type(max_steps) is not int or not 1 <= max_steps <= 1000:
            raise ValueError("max_steps must be an integer between 1 and 1000")
        self.task = TaskSpec.model_validate(task or {})
        self.max_steps = max_steps
        self.reset()

    def reset(self, *, seed=0, options=None):
        options = options or {}
        if set(options) - {"task"}:
            raise ValueError("Only the 'task' reset option is supported")
        task = TaskSpec.model_validate(options.get("task", self.task))
        # No random transitions. Seed is recorded for reproducible experiment/task generators.
        if seed is None:
            seed = 0
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.state = State(task=task, max_steps=self.max_steps, seed=seed)
        self.trajectory = []
        self.setup_actions = []
        self._evaluation = None
        self.initial_observation = self.observe()
        return self.observe(), {"seed": seed}

    def begin_prepared_episode(self):
        """Fixture initialization only: preserve setup separately and start the edit budget.

        Not exposed as an agent action. Used after constructing a reference/repair scene.
        """
        self.setup_actions.extend(copy.deepcopy(t["action"]) for t in self.trajectory)
        self.state = self.state.model_copy(update={"steps": 0, "terminated": False})
        self.trajectory = []
        self._evaluation = None
        self.initial_observation = self.observe()

    def export_trajectory(self):
        """Versioned export also represents an initialized scene with zero user actions."""
        return copy.deepcopy({"format_version": 2, "initial_observation": self.initial_observation,
                              "setup_actions": self.setup_actions, "transitions": self.trajectory})

    def observe(self):
        return observation(self.state)

    def current_reward(self):
        if self._evaluation is None:
            self._evaluation = evaluate(self.state)
        return copy.deepcopy(self._evaluation)

    def step(self, action):
        if self.state.terminated:
            raise EpisodeFinished("Episode finished; reset before editing again")
        before = self.observe()
        elements = list(self.state.elements)
        next_id = self.state.next_id
        error, submitted, result = None, False, {}
        try:
            parsed = ACTION_ADAPTER.validate_python(action)
            if parsed.op == "add_element":
                if len(elements) >= self.state.max_elements:
                    raise ValueError("Element limit reached")
                eid = f"e{next_id}"
                elements.append(Element(**parsed.element.model_dump(), id=eid, creation_index=next_id))
                next_id += 1
                result = {"element_id": eid}
            elif parsed.op == "submit":
                submitted = True
            else:
                index = next((i for i, e in enumerate(elements) if e.id == parsed.id), None)
                if index is None:
                    raise ValueError(f"Unknown element ID: {parsed.id}")
                old = elements[index]
                if parsed.op == "delete_element":
                    elements.pop(index)
                else:
                    changes = {"x": parsed.x, "y": parsed.y} if parsed.op == "move_element" else parsed.properties
                    if set(changes) - set(ElementSpec.model_fields):
                        raise ValueError("Only declared element properties can be changed")
                    elements[index] = Element.model_validate({**old.model_dump(), **changes})
        except (ValueError, TypeError) as exc:
            error = "; ".join(e["msg"] for e in exc.errors(include_url=False, include_input=False)) if isinstance(exc, ValidationError) else str(exc)
            elements, next_id = list(self.state.elements), self.state.next_id
        steps = self.state.steps + 1
        ended = submitted or steps >= self.state.max_steps
        self.state = self.state.model_copy(update={"elements": tuple(elements), "next_id": next_id,
                                                  "steps": steps, "terminated": ended})
        self._evaluation = None
        reward = self.current_reward()["score"] if ended else 0.0
        info = {"action_result": result, "error": error, "end_reason": "submitted" if submitted else "budget" if ended else None}
        if ended:
            info["evaluation"] = self.current_reward()
        after = self.observe()
        self.trajectory.append({"observation": before, "action": copy.deepcopy(action),
                                "next_observation": after, "reward": reward,
                                "terminated": ended, "truncated": False, "info": copy.deepcopy(info)})
        return after, reward, ended, False, info

    def render(self):
        import numpy as np
        return np.asarray(render_scene(self.state, measure=False)[0])

    def save_png(self, path):
        render_scene(self.state, measure=False)[0].save(path)

    def state_hash(self):
        return hashlib.sha256(json.dumps(self.observe(), sort_keys=True).encode()).hexdigest()
