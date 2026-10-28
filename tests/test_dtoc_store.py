"""The DTOC artifact store is bounded: least recently used artifacts are evicted once raw bytes pass the cap."""

import pytest

from harpe import DynamicToolCompressor, LRUArtifactStore
from harpe.dtoc import DEFAULT_STORE_MAX_BYTES


def _blob(n, size=1000):
    return "\n".join(f"line {i} of output {n} " + "x" * 20 for i in range(size // 32))


def test_default_cap_is_64_mib_and_configurable():
    assert DEFAULT_STORE_MAX_BYTES == 64 * 1024 * 1024
    assert DynamicToolCompressor()._store.max_bytes == DEFAULT_STORE_MAX_BYTES
    assert DynamicToolCompressor(store_max_bytes=4096)._store.max_bytes == 4096
    with pytest.raises(ValueError):
        LRUArtifactStore(0)


def test_store_evicts_oldest_once_the_byte_cap_is_passed():
    comp = DynamicToolCompressor(min_compress_chars=100, store_max_bytes=3000)
    refs = []
    for n in range(5):
        _, artifact = comp.maybe_compress("tool", _blob(n))
        assert artifact is not None
        refs.append(artifact.ref_id)
    assert comp.store_bytes <= 3000
    assert comp.store_evictions >= 2
    # The newest survive, the oldest were dropped and restore as None like an unknown ref.
    assert comp.restore(refs[-1]) is not None
    assert comp.restore(refs[0]) is None
    assert comp.restore(refs[1]) is None


def test_a_read_keeps_an_artifact_warm():
    comp = DynamicToolCompressor(min_compress_chars=100, store_max_bytes=2800)  # room for two ~1260-byte blobs
    _, a = comp.maybe_compress("tool", _blob(0))
    _, b = comp.maybe_compress("tool", _blob(1))
    assert comp.restore(a.ref_id) is not None  # touch a: b is now the least recently used
    _, c = comp.maybe_compress("tool", _blob(2))
    assert comp.restore(b.ref_id) is None
    assert comp.restore(a.ref_id) is not None
    assert comp.restore(c.ref_id) is not None


def test_an_artifact_larger_than_the_cap_is_still_restorable_until_the_next_insert():
    comp = DynamicToolCompressor(min_compress_chars=100, store_max_bytes=500)
    _, big = comp.maybe_compress("tool", _blob(0, size=4000))
    assert comp.restore(big.ref_id) is not None
    assert comp.store_bytes > 500
    _, nxt = comp.maybe_compress("tool", _blob(1))
    assert comp.restore(big.ref_id) is None
    assert comp.restore(nxt.ref_id) is not None


def test_byte_accounting_survives_replace_pop_and_clear():
    store = LRUArtifactStore(10_000)
    comp = DynamicToolCompressor(min_compress_chars=100)
    comp._store = store
    _, a = comp.maybe_compress("tool", _blob(0))
    size = a.raw_byte_size
    assert store.total_bytes == size
    store[a.ref_id] = a  # replacing the same key does not double count
    assert store.total_bytes == size
    assert store.pop(a.ref_id) is a and store.total_bytes == 0
    assert store.pop("missing", "dflt") == "dflt"
    comp.maybe_compress("tool", _blob(1))
    store.clear()
    assert store.total_bytes == 0 and len(store) == 0
