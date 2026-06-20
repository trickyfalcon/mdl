"""Tests for download status: the pure classifier + the cache-scanning wrappers."""

from mdl import hub
from mdl.config import Config
from mdl.hub import classify_download
from mdl.paths import free_space


# -- classify_download (pure) -------------------------------------------------------------
def test_classify_missing_unverified():
    s = classify_download(0, 0, 0, None, None)
    assert s.state == "missing" and s.verified is False and s.remaining_bytes is None


def test_classify_missing_verified():
    s = classify_download(0, 0, 0, 1000, 5)
    assert s.state == "missing" and s.verified is True


def test_classify_complete_exact():
    s = classify_download(1000, 5, 0, 1000, 5)
    assert s.state == "complete" and s.remaining_bytes == 0


def test_classify_complete_within_tolerance():
    # 0.1% short still counts as complete (size metadata is approximate)
    s = classify_download(999, 5, 0, 1000, 5)
    assert s.state == "complete"


def test_classify_partial_when_incomplete_present():
    s = classify_download(1000, 5, 2, 1000, 5)
    assert s.state == "partial"


def test_classify_partial_when_files_missing():
    s = classify_download(500, 3, 0, 1000, 5)
    assert s.state == "partial" and s.remaining_bytes == 500


def test_classify_present_but_unverified_is_partial_never_complete():
    # offline (no Hub metadata): anything on disk must re-run hf, never silently skip
    s = classify_download(9_000, 3, 0, None, None)
    assert s.state == "partial" and s.verified is False


# -- raw_status (scans the HF cache) ------------------------------------------------------
def _cfg_with_home(tmp_path):
    cfg = Config(values=dict(Config().values))
    cfg.values["hf_home"] = str(tmp_path)
    return cfg


def _make_raw_cache(tmp_path, repo, *, shard_bytes, n_shards, incomplete=0):
    base = tmp_path / "hub" / ("models--" + repo.replace("/", "--"))
    snap = base / "snapshots" / "rev0"
    snap.mkdir(parents=True)
    for i in range(n_shards):
        (snap / f"model-{i:05d}.safetensors").write_bytes(b"x" * shard_bytes)
    blobs = base / "blobs"
    blobs.mkdir(parents=True)
    for i in range(incomplete):
        (blobs / f"deadbeef{i}.abc.incomplete").write_bytes(b"y" * 10)


def test_raw_status_complete(tmp_path, monkeypatch):
    cfg = _cfg_with_home(tmp_path)
    _make_raw_cache(tmp_path, "owner/Model", shard_bytes=100, n_shards=4)
    monkeypatch.setattr(hub, "_repo_sizes", lambda repo, predicate=None: (400, 4))
    s = hub.raw_status(cfg, "owner/Model")
    assert s.state == "complete" and s.present_files == 4 and s.present_bytes == 400


def test_raw_status_partial_with_incomplete(tmp_path, monkeypatch):
    cfg = _cfg_with_home(tmp_path)
    _make_raw_cache(tmp_path, "owner/Model", shard_bytes=100, n_shards=2, incomplete=3)
    monkeypatch.setattr(hub, "_repo_sizes", lambda repo, predicate=None: (500, 5))
    s = hub.raw_status(cfg, "owner/Model")
    assert s.state == "partial" and s.incomplete == 3 and s.remaining_bytes == 300


def test_raw_status_missing(tmp_path, monkeypatch):
    cfg = _cfg_with_home(tmp_path)
    monkeypatch.setattr(hub, "_repo_sizes", lambda repo, predicate=None: (400, 4))
    s = hub.raw_status(cfg, "owner/Nope")
    assert s.state == "missing"


def test_raw_status_offline_present_is_partial(tmp_path, monkeypatch):
    cfg = _cfg_with_home(tmp_path)
    _make_raw_cache(tmp_path, "owner/Model", shard_bytes=100, n_shards=4)
    monkeypatch.setattr(hub, "_repo_sizes", lambda repo, predicate=None: (None, None))
    s = hub.raw_status(cfg, "owner/Model")
    assert s.state == "partial" and s.verified is False


# -- gguf_status (scans the local-dir, filters by quant) ----------------------------------
def test_gguf_status_complete_for_quant(tmp_path, monkeypatch):
    cfg = _cfg_with_home(tmp_path)
    target = tmp_path / "owner" / "Model-GGUF"
    target.mkdir(parents=True)
    (target / "Model-Q4_K_M.gguf").write_bytes(b"x" * 200)
    (target / "Model-Q8_0.gguf").write_bytes(b"x" * 999)  # other quant, ignored
    monkeypatch.setattr(
        hub, "_repo_sizes",
        lambda repo, predicate=None: (200, 1) if predicate("Model-Q4_K_M.gguf") else (0, 0),
    )
    s = hub.gguf_status(cfg, "owner/Model-GGUF", "Q4_K_M", target)
    assert s.state == "complete" and s.present_bytes == 200


# -- free_space ---------------------------------------------------------------------------
def test_free_space_existing_path_returns_int(tmp_path):
    free = free_space(tmp_path)
    assert isinstance(free, int) and free >= 0


def test_free_space_nonexistent_subdir_walks_up(tmp_path):
    # a not-yet-created target dir still resolves to its mounted drive's free space
    free = free_space(tmp_path / "does" / "not" / "exist" / "yet")
    assert isinstance(free, int) and free >= 0
