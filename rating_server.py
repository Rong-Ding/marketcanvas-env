"""Independent local rating server; leaves the canvas inspector and its state alone."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from viewer_analysis import write_report
from viewer_pilot import DEFAULT_OUTPUT, ROOT, sha, verify_study


def now():
    return datetime.now(timezone.utc).isoformat()


class RatingStore:
    def __init__(self, study, response_path):
        self.study, self.path = Path(study), Path(response_path)
        self.protocol = verify_study(self.study)
        self.manifest = json.loads((self.study / "public_manifest.json").read_text())
        self.freeze_hash = (self.study / "freeze.sha256").read_text().strip()
        self.ids = {p["id"] for p in self.manifest["pairs"]}
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.write({"study_id": self.manifest["study_id"], "freeze_hash": self.freeze_hash,
                        "reviewer_id": "reviewer_01", "revision": 0, "ratings": {}, "completed_at": None})
        self.read()

    def read(self):
        result = json.loads(self.path.read_text())
        if result["freeze_hash"] != self.freeze_hash:
            raise ValueError("These responses belong to a different frozen study")
        return result

    def write(self, record):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n")
        temp.replace(self.path)

    def save(self, data):
        with self.lock:
            record = self.read()
            if record["completed_at"]:
                raise ValueError("Ratings are locked after revealing results")
            if type(data.get("revision")) is not int or data["revision"] != record["revision"]:
                raise ValueError("Another tab changed these ratings. Reload before continuing.")
            if set(data) != {"pair_id", "discoverability", "preference", "notes", "revision"}:
                raise ValueError("Unexpected rating fields")
            if data["pair_id"] not in self.ids:
                raise ValueError("Unknown pair")
            for key in ("discoverability", "preference"):
                if data[key] not in (None, "left", "right", "tie", "unsure"):
                    raise ValueError("Invalid rating choice")
            if not isinstance(data["notes"], str) or len(data["notes"]) > 1000:
                raise ValueError("Notes must be at most 1000 characters")
            record["ratings"][data["pair_id"]] = {k: data[k] for k in ("discoverability", "preference", "notes")}
            record["ratings"][data["pair_id"]]["saved_at"] = now()
            record["revision"] += 1
            self.write(record)
            return record

    def finish(self, revision):
        with self.lock:
            record = self.read()
            if record["completed_at"]:
                return record
            if type(revision) is not int or revision != record["revision"]:
                raise ValueError("Ratings changed; reload before finishing")
            if any(not all(record["ratings"].get(pair, {}).get(q) for q in ("discoverability", "preference")) for pair in self.ids):
                raise ValueError("Answer both questions for every pair; unsure is a valid answer")
            record["completed_at"] = now()
            record["revision"] += 1
            self.write(record)
            return record


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def send(self, value, kind="application/json", status=200):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self.send((store.study / "frozen_source/marketcanvas/web/rating.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/study":
                return self.send({"manifest": store.manifest, "freeze_hash": store.freeze_hash, "record": store.read()})
            if path == "/api/ratings":
                return self.send(store.read())
            allowed = {"/"+p[side] for p in store.manifest["pairs"] for side in ("left", "right")}
            if path in allowed:
                return self.send((store.study / path.lstrip("/")).read_bytes(), "image/png")
            if path == "/api/results":
                try:
                    result = write_report(store.study, store.read(), store.path.parent / "analysis")
                    return self.send(result)
                except ValueError as exc:
                    return self.send({"error": str(exc)}, status=403)
            return self.send({"error": "Not found"}, status=404)

        def do_POST(self):
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Origin") != expected:
                return self.send({"error": "Same-origin requests required"}, status=403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send({"error": "JSON required"}, status=415)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8192:
                    raise ValueError("Request too large or empty")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected an object")
                if self.path == "/api/rate":
                    return self.send(store.save(data))
                if self.path == "/api/finish":
                    record = store.finish(data.get("revision"))
                    return self.send({"record": record, "results": write_report(store.study, record, store.path.parent / "analysis")})
                return self.send({"error": "Not found"}, status=404)
            except (ValueError, TypeError, KeyError) as exc:
                return self.send({"error": str(exc)}, status=400)

        def log_message(self, *_):
            pass
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--responses", type=Path, default=ROOT / "private/viewer-pilot-v1/ratings.json")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    store = RatingStore(args.study, args.responses)
    # Fail rather than silently analyze frozen predictions using altered live code.
    for name in ("viewer_analysis.py", "rating_server.py"):
        if sha(ROOT / name) != sha(args.study / "frozen_source" / name):
            raise ValueError(f"{name} differs from frozen version; create a new study version")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store))
    print(f"Viewer pilot: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()
