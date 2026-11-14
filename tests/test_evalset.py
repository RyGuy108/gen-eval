import shutil

import pytest

from geneval import evalset


def test_v1_loads_and_is_frozen():
    es = evalset.load("v1")
    assert len(es.items) == len({i.id for i in es.items})
    assert {i.suite for i in es.items} == {"text", "adherence", "character", "video"}
    for it in es.items:
        if it.suite in ("character", "video"):
            assert es.ref_path(it).exists()
        if it.suite == "adherence":
            assert len(it.foils) >= 2


def test_editing_a_frozen_set_is_refused(tmp_path):
    shutil.copytree(evalset.ROOT / "v1", tmp_path / "v1")
    p = tmp_path / "v1" / "suites.yaml"
    p.write_text(p.read_text().replace("three red apples", "four red apples", 1))
    with pytest.raises(RuntimeError, match="edited after freezing"):
        evalset.load("v1", root=tmp_path)


def test_swapping_a_reference_image_is_detected(tmp_path):
    shutil.copytree(evalset.ROOT / "v1", tmp_path / "v1")
    refs = tmp_path / "v1" / "refs"
    (refs / "maya.jpg").write_bytes((refs / "dele.jpg").read_bytes())
    with pytest.raises(RuntimeError, match="c01"):
        evalset.load("v1", root=tmp_path)


def test_unchanged_items_keep_their_hash_across_versions(tmp_path):
    shutil.copytree(evalset.ROOT / "v1", tmp_path / "v1")
    p = tmp_path / "v1" / "suites.yaml"
    p.write_text(p.read_text().replace("three red apples", "four red apples", 1))
    a = evalset.load("v1").by_id()
    b = evalset.load("v1", root=tmp_path, verify=False).by_id()
    assert a["a01"].hash != b["a01"].hash
    assert a["t01"].hash == b["t01"].hash
