"""Local-only inspector. python playground.py; open http://127.0.0.1:8765.

The HTTP inspector and stdio MCP each own independent episodes of the same core.
This is a development tool, not a production HTTP service.
"""
import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from marketcanvas.env import MarketCanvasEnv, EpisodeFinished
from marketcanvas.models import TaskSpec
from marketcanvas.scenarios import load_scenario, CASES

ROOT = Path(__file__).parent
env = load_scenario()
lock = threading.RLock()


class Handler(BaseHTTPRequestHandler):
    def send(self, body, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def payload(self):
        return {"observation": env.observe(), "evaluation": env.current_reward(), "scenarios": CASES}

    def do_GET(self):
        with lock:
            if self.path == "/":
                return self.send((ROOT / "marketcanvas/web/index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self.send(json.dumps(self.payload()).encode())
            if self.path.startswith("/canvas.png"):
                from PIL import Image
                buffer = io.BytesIO()
                Image.fromarray(env.render()).save(buffer, format="PNG")
                return self.send(buffer.getvalue(), "image/png")
            if self.path == "/api/trajectory":
                return self.send(json.dumps(env.export_trajectory(), indent=2).encode())
            self.send(b'{"error":"Not found"}', status=404)

    def do_POST(self):
        global env
        # Require same-origin JSON; do not permit arbitrary web pages to mutate localhost.
        expected = f"http://127.0.0.1:{self.server.server_port}"
        origin = self.headers.get("Origin")
        if origin and origin != expected:
            return self.send(b'{"error":"Origin rejected"}', status=403)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.send(b'{"error":"JSON required"}', status=415)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16384:
                raise ValueError("Request must contain 1-16384 bytes")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must be an object")
            with lock:
                if self.path == "/api/action":
                    result = env.step(data)
                    return self.send(json.dumps({**self.payload(), "step_result": {"reward": result[1], "info": result[4]}}).encode())
                if self.path == "/api/reset":
                    task = TaskSpec.model_validate(data.get("task", {}))
                    seed = data.get("seed", 0)
                    scenario = data.get("scenario", "blank")
                    if scenario not in CASES + ["blank"]:
                        raise ValueError("Unknown scenario")
                    if scenario == "blank":
                        candidate = MarketCanvasEnv(task=task)
                        candidate.reset(seed=seed)
                    else:
                        candidate = load_scenario(scenario, seed, task)
                    env = candidate
                    return self.send(json.dumps(self.payload()).encode())
                self.send(b'{"error":"Not found"}', status=404)
        except (ValueError, TypeError, EpisodeFinished) as exc:
            self.send(json.dumps({"error": str(exc)}).encode(), status=400)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MarketCanvas inspector: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
