"""The speed-ups taken from the third review of PR #16 (review_notes/R3_PR16.md, Task D) change no result.

Each test compares the code in ``src/harpe`` with the definition it had before the speed-up, written out here.
"""
from __future__ import annotations

import random
import re
from typing import List

from harpe.engine import HeuristicTokenCounter

_PIECE_RE = re.compile(r"[A-Za-z0-9_]+|\S")
_HEX_RUN_RE = re.compile(r"[0-9a-fA-F]{8,}")


def _count_in_two_passes(text: str) -> int:
    """``HeuristicTokenCounter.count_tokens`` as it was: pieces over the text, then hex runs over the text."""
    if not text:
        return 0
    pieces = len(_PIECE_RE.findall(text))
    pieces += sum(len(run) // 3 for run in _HEX_RUN_RE.findall(text))
    return max(pieces, (len(text) + 3) // 4)


def _texts() -> List[str]:
    rng = random.Random(20261110)
    parts = ["a", "f", "F", "0", "9", "g", "G", "_", " ", "\n", "\t", "-", ":", "\u00e9", "\u6f22", "\U0001F600",
             "\u0661", "\uff11", "\uff41", "deadbeef", "DEADBEEF", "0123456789abcdef", "x" * 9, "cafe", "abcdef0", "\u2028"]
    texts = ["".join(rng.choice(parts) for _ in range(rng.randint(0, 40))) for _ in range(20000)]
    texts += [
        "", " ", "deadbeef", "deadbee", "deadbeefg", "gdeadbeef", "dead_beef", "deadbeef_deadbeef", "deadbeef-deadbeef",
        "0123456789abcdef0123456789abcdef", "sha256:" + "ab12" * 16, "x" * 7 + "deadbeef", "deadbeef" + "x" * 7,
        "commit 9fceb02d0ae598e95dc970b74767f19372d61af8 touched 12 files", "\u00e9deadbeef\u00e9", "deadbeef\u6f22cafebabe",
        "\uff41\uff42\uff43\uff44\uff45\uff46\uff10\uff11",  # full-width letters and digits: not ASCII, so not a hex run
        "word " * 500, "0123456789abcdef " * 200, ("f" * 8 + "g") * 100,
    ]
    return texts


def test_count_tokens_equals_the_two_pass_definition():
    count = HeuristicTokenCounter().count_tokens
    for text in _texts():
        assert count(text) == _count_in_two_passes(text), repr(text)
