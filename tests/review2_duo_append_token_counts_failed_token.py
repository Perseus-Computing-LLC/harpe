"""severity: low
finding: DuoAttentionKVCacheManager.append_token() counts the generated token before sizing it, so an append whose sizing raises still advances generated_tokens, and the natural retry counts the token twice

The round-1 fix reordered ``allocate_sequence`` (src/harpe/compression_vllm.py:433-437) so that "a prompt length
the sizing rejects leaves no half-registered sequence behind", and ``tests/test_crash_consistency.py:173-182``
holds it to that by making ``compute_sequence_vram_bytes`` raise. The sibling write path in the same class was not
fixed: ``append_token`` (:439-444) does ``self.sequences[seq_id]["generated_tokens"] += 1`` (:442) first and only
then calls ``compute_sequence_vram_bytes`` (:444). When sizing raises (simulated here exactly as the project's own
test does; naturally reachable through the ``int(...)`` of an overflowing float at :423/:425), the caller is told
the append failed, but the sequence already accounts for the token. A retry of the failed append then leaves
``generated_tokens == 2`` for one token, and every later sizing over-reports the sequence's VRAM.
"""
from __future__ import annotations

import pytest

from harpe.compression_vllm import DuoAttentionKVCacheManager


def _failing(_seq_len):
    raise ValueError("simulated: sizing failed mid-append")


def test_control_a_clean_append_counts_one_token():
    mgr = DuoAttentionKVCacheManager(num_layers=2, num_heads=8, head_dim=64)
    mgr.allocate_sequence("s1", 128)
    sizing = mgr.append_token("s1")
    assert mgr.sequences["s1"]["generated_tokens"] == 1 and sizing["seq_len"] == 129


def test_an_append_whose_sizing_raises_counts_no_token(monkeypatch):
    mgr = DuoAttentionKVCacheManager(num_layers=2, num_heads=8, head_dim=64)
    mgr.allocate_sequence("s1", 128)
    before = dict(mgr.sequences["s1"])

    monkeypatch.setattr(mgr, "compute_sequence_vram_bytes", _failing)
    with pytest.raises(ValueError):
        mgr.append_token("s1")
    monkeypatch.undo()

    assert mgr.sequences["s1"] == before, "a failed append_token still advanced generated_tokens"
    sizing = mgr.append_token("s1")  # the caller retries the append that failed
    assert mgr.sequences["s1"]["generated_tokens"] == 1 and sizing["seq_len"] == 129, (
        "the retried append counted one token twice"
    )


def test_an_append_whose_real_sizing_overflows_counts_no_token():
    """No monkeypatch: the prompt is sized fine, the prompt plus one token overflows the float and int(inf) raises."""
    mgr = DuoAttentionKVCacheManager(num_layers=1, num_heads=1, head_dim=1)
    prompt = 2 ** 1022 - 2 ** 968 - 1  # 2*prompt*2.0 is DBL_MAX; 2*(prompt+1) rounds to 2**1023, *2.0 is inf
    mgr.allocate_sequence("s1", prompt)
    with pytest.raises(OverflowError):
        mgr.append_token("s1")
    assert mgr.sequences["s1"]["generated_tokens"] == 0, "a failed append_token still advanced generated_tokens"
