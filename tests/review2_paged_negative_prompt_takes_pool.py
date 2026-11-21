"""severity: low
finding: the round-1 rewrite of PagedKVCacheManager.allocate_sequence slices free_blocks[:needed_blocks]; a negative prompt_tokens makes needed_blocks negative, so the 'plan' takes every free block but the last, records a negative token count, and the next admission fails with MemoryError

Regression introduced by the fix diff (``git diff f6b32d4 HEAD -- src/harpe/compression_vllm.py``). Before it, the
allocation loop was ``for _ in range(needed_blocks)``, which takes no block when ``needed_blocks`` is <= 0. The fix
plans with ``for b_id in self.free_blocks[:needed_blocks]`` (src/harpe/compression_vllm.py:211) and commits with
``del self.free_blocks[:needed_blocks]`` (:217). ``needed_blocks = math.ceil(prompt_tokens / block_size)`` (:201) is
-1 for ``prompt_tokens=-20``, and the capacity check ``len(free) < needed`` (:202) passes, so the slice
``[:-1]`` is "every free block but the last". On a 10-block pool one call with ``prompt_tokens=-20`` takes 9 blocks,
stores ``num_tokens_stored=-20`` in the first (:220, ``PhysicalBlock`` does not validate assignment), reports
utilization -12.5% and internal fragmentation 113.89%, and a following 32-token sequence is refused with
``MemoryError``. The fix's own comment says "a failure mid-plan leaks no block" (:208); this input leaks almost the
whole pool without any failure. The test accepts either refusing the prompt or taking no block, which is what the
pre-fix code did.
"""
from __future__ import annotations

from harpe.compression_vllm import PagedKVCacheManager


def _mgr() -> PagedKVCacheManager:
    return PagedKVCacheManager(total_vram_bytes_for_kv=16 * 64 * 10, bytes_per_token=64, block_size_tokens=16)


def test_control_a_positive_prompt_takes_exactly_its_blocks():
    mgr = _mgr()
    assert mgr.allocate_sequence("s1", 40) == [0, 1, 2]
    assert mgr.free_blocks == list(range(3, 10))


def test_a_negative_prompt_takes_no_block():
    mgr = _mgr()
    free = list(mgr.free_blocks)
    try:
        mgr.allocate_sequence("bad", -20)
    except (ValueError, MemoryError):
        pass  # refusing the prompt is fine
    assert mgr.free_blocks == free, (
        f"a prompt of -20 tokens took {len(free) - len(mgr.free_blocks)} of {len(free)} blocks"
    )
    assert all(b.num_tokens_stored >= 0 for b in mgr.physical_blocks.values())
    mgr.allocate_sequence("next", 32)  # the pool is still there for the next sequence
