import json
from dataclasses import replace
from http.server import ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from marketcanvas.scenarios import load_scenario, apply
from marketcanvas.viewer import ViewerConfig, evaluate_viewer, extract_regions, predict_visits
from rating_server import RatingStore, handler_for
from viewer_analysis import analyze
from viewer_pilot import build_study, make_scene, verify_study


def test_uniform_viewer_matches_closed_form_and_preserves_probability_mass():
    regions = [{"id": str(i), "size_normalized": .2, "contrast_normalized": .4, "x": .5, "y": .5} for i in range(4)]
    result = predict_visits(regions, ViewerConfig(beta=0, distance_weight=0, revisit_penalty=0))
    assert result["mass_by_glance"] == pytest.approx([1, 1, 1])
    for probs in result["visit_probability"].values():
        assert probs == pytest.approx([1-(3/4)**k for k in (1, 2, 3)])


def test_viewer_role_blindness_and_reordering():
    env = load_scenario()
    original = extract_regions(env.state)
    relabeled = env.state.model_copy(update={"elements": tuple(e.model_copy(update={"role": "decoration"}) for e in env.state.elements)})
    assert extract_regions(relabeled) == original
    forward = predict_visits(original)
    reverse = predict_visits(list(reversed(original)))
    for key, value in forward["visit_probability"].items():
        assert reverse["visit_probability"][key] == pytest.approx(value)


def test_occluded_regions_are_excluded_and_essential_failures_stay_negative():
    env = load_scenario("covered_headline")
    assert "e1" not in {r["id"] for r in extract_regions(env.state)}
    for case in ("covered_headline", "missing_cta", "low_contrast", "duplicate_cta"):
        assert evaluate_viewer(load_scenario(case).state)["score"] < 0


def test_readonly_deterministic_and_alpha_zero_recovers_baseline():
    env = load_scenario()
    before = env.state_hash(), env.export_trajectory()
    one = evaluate_viewer(env.state)
    assert evaluate_viewer(env.state) == one
    assert 0 <= one["D1"] <= one["D2"] <= one["D3"] <= 1+1e-12
    assert evaluate_viewer(env.state, alpha=0)["score"] == env.current_reward()["score"]
    assert (env.state_hash(), env.export_trajectory()) == before


def test_memory_changes_later_visits_not_initial_glance():
    regions = extract_regions(load_scenario().state)
    a = predict_visits(regions, ViewerConfig(revisit_penalty=0))
    b = predict_visits(regions, ViewerConfig(revisit_penalty=2))
    assert a["first_glance"] == b["first_glance"]
    assert sum(v[1] for v in b["visit_probability"].values()) > sum(v[1] for v in a["visit_probability"].values())


@pytest.mark.parametrize("kw", [{"beta": float('nan')}, {"start_x": 2}, {"revisit_penalty": -1}])
def test_config_rejects_invalid_settings(kw):
    with pytest.raises(ValueError):
        ViewerConfig(**kw)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("viewer") / "study"
    build_study(directory)
    return directory


def test_frozen_pack_has_valid_paired_stimuli_and_no_public_predictions(study):
    protocol = verify_study(study)
    assert build_study(study) == protocol
    manifest = json.loads((study / "public_manifest.json").read_text())
    assert len(manifest["pairs"]) == 12
    assert "score" not in json.dumps(manifest)
    predictions = json.loads((study / "predictions.json").read_text())
    assert sum(p["swapped"] for p in predictions.values()) == 6
    for pair in predictions.values():
        for side in ("left", "right"):
            assert pair[side]["primary"]["baseline_score"] == 1
            assert pair[side]["primary"]["success"]


def test_partial_ratings_resume_revision_control_and_completion_lock(study, tmp_path):
    path = tmp_path / "ratings.json"
    store = RatingStore(study, path)
    assert not store.read()["ratings"]
    with pytest.raises(ValueError):
        analyze(study, store.read())
    partial = dict(pair_id="P01", discoverability="left", preference=None, notes="test only", revision=0)
    first = store.save(partial)
    assert RatingStore(study, path).read() == first
    with pytest.raises(ValueError):
        store.save(partial)
    with pytest.raises(ValueError):
        store.finish(first["revision"])
    for pair in store.manifest["pairs"]:
        current = store.read()
        store.save(dict(pair_id=pair["id"], discoverability="unsure", preference="tie", notes="", revision=current["revision"]))
    done = store.finish(store.read()["revision"])
    result = analyze(study, done)
    unsure = next(r for r in result["summaries"] if r["model"] == "viewer" and r["question"] == "discoverability")
    assert unsure["agreement_fraction"] is None and unsure["unsure"] == 12
    ties = next(r for r in result["summaries"] if r["model"] == "original" and r["question"] == "preference")
    assert ties["agreements"] == ties["human_ties"] == 12
    with pytest.raises(ValueError):
        store.save({**partial, "revision": done["revision"]})


def test_public_server_hides_predictions_and_rejects_cross_origin(study, tmp_path):
    store = RatingStore(study, tmp_path / "ratings.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/study") as response:
            assert "baseline_score" not in response.read().decode()
        for path, status in (("/predictions.json", 404), ("/protocol.json", 404), ("/api/results", 403), ("/stimuli/../predictions.json", 404)):
            with pytest.raises(HTTPError) as exc:
                urlopen(base + path)
            assert exc.value.code == status
        request = Request(base + "/api/rate", data=b'{}', headers={"Content-Type": "application/json", "Origin": "https://example.org"})
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 403
        assert store.read()["ratings"] == {}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
