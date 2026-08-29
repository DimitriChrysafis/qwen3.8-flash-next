from colibri_runtime import ExpertCache, NgramDiskCache


def test_expert_cache_eviction_and_pinning() -> None:
    c = ExpertCache(max_items=2)
    c.put("a", 1)
    c.put("b", 2, pin=True)
    c.put("c", 3)
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3
    snap = c.snapshot()
    assert snap["evictions"] >= 1
    assert snap["pinned"] == 1


def test_ngram_disk_cache(tmp_path) -> None:
    db = NgramDiskCache(str(tmp_path / "ngrams.sqlite"))
    assert db.get("x") is None
    db.put("x", b"abc")
    assert db.get("x") == b"abc"
    assert db.count() == 1
