import math

from geneval import quality, rank


def test_text_pass_requires_every_string():
    m = {"ocr": {"legibility": 0.98, "per_target": [1.0, 0.96], "exact": False, "gibberish": 0}}
    q, ok = quality.judge("text", m)
    assert q == 0.96 and ok
    m["ocr"]["per_target"] = [1.0, 0.9]
    assert not quality.judge("text", m)[1]


def test_text_invented_gibberish_fails():
    m = {"ocr": {"per_target": [1.0], "gibberish": 4}}
    q, ok = quality.judge("text", m)
    assert not ok and q == 0.7


def test_character_returning_reference_fails():
    # perfect identity but the studio-headshot foil wins -> not a pass, low quality
    m = {"face": {"face_sim": 0.99, "face_sim_min": 0.99}, "clip": {"contrastive": 0.05, "beats_all_foils": False}}
    q, ok = quality.judge("character", m)
    assert not ok and q < 0.3


def test_video_uses_worst_frame():
    m = {"face": {"face_sim": 0.7, "face_sim_min": 0.2}, "clip": {"contrastive": 0.9, "beats_all_foils": True}}
    assert not quality.judge("video", m)[1]


def test_bars_are_nested():
    m = {"clip": {"contrastive": 0.8, "beats_all_foils": True, "margin": 0.05}}
    b = quality.bars("adherence", m)
    assert b == {"draft": 1, "standard": 1, "premium": 1}
    m["clip"]["margin"] = 0.01
    assert quality.bars("adherence", m)["premium"] == 0


def test_pareto():
    pts = {"cheap_bad": (5, 0.3), "mid": (50, 0.8), "dominated": (60, 0.7), "pro": (150, 0.9)}
    assert rank.pareto(pts) == {"cheap_bad", "mid", "pro"}


def _rows(model, credits, passes):
    return [{"suite": "text", "model": model, "item_id": f"t{i}", "credits": credits, "pass": p, "q": float(p)}
            for i, p in enumerate(passes)]


def test_credits_per_pass_penalizes_cheap_failures():
    rows = _rows("cheap", 5, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) + _rows("solid", 30, [1] * 9 + [0])
    s = rank.summarize(rows)["text"]
    by = {m["model"]: m for m in s}
    assert by["cheap"]["credits_per_pass"] == 50
    assert math.isclose(by["solid"]["credits_per_pass"], 30 / 0.9)
    assert s[0]["model"] == "solid"
    lo, hi = by["solid"]["pass_ci"]
    assert lo <= 0.9 <= hi


def test_zero_pass_rate_ranks_last():
    rows = _rows("never", 1, [0, 0, 0]) + _rows("ok", 100, [1, 0, 1])
    s = rank.summarize(rows)["text"]
    assert s[-1]["model"] == "never" and math.isinf(s[-1]["credits_per_pass"])


def test_router_bars_for_every_suite():
    face = {"face_sim": 0.55, "face_sim_min": 0.5}
    clip = {"contrastive": 0.9, "beats_all_foils": True, "margin": 0.05}
    assert quality.bars("character", {"face": face, "clip": clip}) == {"draft": 1, "standard": 1, "premium": 1}
    assert quality.bars("video", {"face": face, "clip": clip})["premium"] == 1
    checks = {"fact_score": 0.5, "all_facts": False}
    assert quality.bars("adherence", {"clip": clip, "checks": checks}) == {"draft": 1, "standard": 0, "premium": 0}
