"""$0 offline verification of the ProfileLedger against the banked MemConflict Round 5 corpus.

No model, no GPU, no network when the corpus is cached. Replays the harness ledger path over all 3,750
MemConflict questions (tests/memconflict_replay.py) and compares each block's SHA-256 with the banked 0.1.0a8
digests, which were taken from blocks verified byte-identical to the Round 5 H100 run (Retrieved_Memories[0] of
every question whose retrieval carried profile_tokens; 3,294 with a block, 456 unrouted). Only digests are
banked - the blocks quote third-party dialogue, which stays out of this repository.

The contract is explicit: every question outside CHANGED is byte-identical to 0.1.0a8, every question in CHANGED
differs, and the named target cases carry the expected fix. Any other drift fails the test. When a deliberate
ledger change moves a question, add it to CHANGED with the reason - the diff tool prints the list:

    git worktree add /tmp/a8 v0.1.0-alpha.8 && (cd /tmp/a8 && python -m tests.memconflict_replay \
        ../harpe/tests/data/Step4_4.jsonl /tmp/a8.jsonl)      # the previous build's blocks, with text
    python -m tests.memconflict_replay tests/data/Step4_4.jsonl /tmp/new.jsonl
    python -m tests.memconflict_diff /tmp/a8.jsonl /tmp/new.jsonl

Corpus: tests/data/Step4_4.jsonl (gitignored, 40 MB) or $HARPE_MEMCONFLICT_CORPUS; fetched once from the pinned
MemConflict commit when absent and the network allows. SHA-256 is checked either way; the test skips, not fails,
when the corpus is unavailable, and never when it is wrong.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.memconflict_replay import replay  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
CORPUS_SHA256 = "8ef9ec8589eccb86f63ab3a819a9180217405351a8d5846866721ea74babe092"
CORPUS_URL = ("https://raw.githubusercontent.com/TaoZhen1110/MemConflict/"
              "ec51d5d36e87f7665d1337f3a88cbde95fc2a964/Data/Step4_4.jsonl")
BASELINE = DATA / "memconflict_r5_profile_blocks_a8.sha256.jsonl"
BASELINE_SHA256 = "c6fc0538b06d8d3948223838a9fc69bf0d5386895e3fbd0f82c185d3e0b791af"
TOTAL = 3750

Key = Tuple[str, int, str]

# Questions whose block legitimately differs from 0.1.0a8, each a static_conflict family question whose gold
# answer says the sources disagree. Persona prefix (8 hex), session, question id.
CHANGED = {
    # Item 5: a kinship noun owned by someone else is not the user's ("her father", "the father" of a child).
    ("5c2a85a4", 33, "Q_001"),  # target case: record becomes "I don't have a father to rely on", not "her father"
    ("cafa273c", 47, "Q_001"),  # "shared only with the father" (a child's) dropped from the father record
    ("3c2e5fe5", 45, "Q_001"),  # "the mom call" (with Alex) dropped; another has:yes line takes the conflict slot
    ("4cbe1c3d", 49, "Q_001"),  # adoption-records sentence no longer read as the user's mother statement
    # Item 6: has:no after inferred has:yes -> LATER CONFLICTING CLAIM (negations the a8 patterns did not read:
    # "don't really have siblings", "no known/recorded/maternal ...", "she passed away", "raised solely by ...").
    ("14e41890", 47, "Q_001"), ("40a79d0c", 32, "Q_001"), ("40a79d0c", 45, "Q_001"), ("4cbe1c3d", 42, "Q_001"),
    ("5c2a85a4", 48, "Q_001"), ("737e8d59", 46, "Q_001"), ("75345e85", 14, "Q_001"), ("75345e85", 45, "Q_001"),
    ("8518c87c", 52, "Q_001"), ("9841f645", 14, "Q_001"), ("9dffc5de", 28, "Q_001"), ("9dffc5de", 42, "Q_001"),
    ("bb30a4e6", 50, "Q_001"), ("bb30a4e6", 52, "Q_001"), ("c2efaebe", 49, "Q_001"), ("cafa273c", 43, "Q_001"),
    ("cafa273c", 48, "Q_001"), ("d65aae82", 41, "Q_001"), ("f398d596", 41, "Q_001"), ("f47bb65a", 45, "Q_001"),
    ("f5f6c655", 44, "Q_001"), ("2e62aacf", 18, "Q_001"), ("e57c356a", 31, "Q_001"), ("fc1726f1", 37, "Q_001"),
    # Reported negatives ("a friend thought I had no siblings") no longer read as the user's own has:no.
    ("4cbe1c3d", 34, "Q_001"), ("4cbe1c3d", 46, "Q_001"), ("8518c87c", 28, "Q_001"), ("8518c87c", 46, "Q_001"),
    ("9dffc5de", 35, "Q_001"), ("f47bb65a", 14, "Q_001"),
}
assert len(CHANGED) == 34


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _corpus() -> Path:
    path = Path(os.environ.get("HARPE_MEMCONFLICT_CORPUS", DATA / "Step4_4.jsonl"))
    if not path.exists():
        try:
            DATA.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(CORPUS_URL, path)  # noqa: S310 - pinned commit, SHA checked below
        except Exception as exc:  # pragma: no cover - network dependent
            pytest.skip(f"MemConflict corpus not available offline ({exc}); set HARPE_MEMCONFLICT_CORPUS")
    digest = _sha256(path)
    assert digest == CORPUS_SHA256, f"corpus SHA-256 {digest} != pinned {CORPUS_SHA256}"
    return path


def _digest(block: str) -> str:
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def _baseline() -> Dict[Key, dict]:
    """Banked a8 rows: attributes, block_sha256, block_len (no block text)."""
    assert _sha256(BASELINE) == BASELINE_SHA256, "banked a8 baseline changed"
    out: Dict[Key, dict] = {}
    with open(BASELINE, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(r["persona"], r["session"], r["question_id"])] = r
    return out


def _same(base_row: dict, new_row: dict) -> bool:
    return (base_row["attributes"] == new_row["attributes"]
            and base_row["block_sha256"] == _digest(new_row["block"]))


@pytest.fixture(scope="module")
def replayed() -> Tuple[Dict[Key, dict], Dict[Key, dict]]:
    base = _baseline()
    new: Dict[Key, dict] = {}
    for r in replay(str(_corpus())):
        r["block_sha256"] = _digest(r["block"])
        new[(r["persona"], r["session"], r["question_id"])] = r
    assert len(base) == len(new) == TOTAL
    assert set(base) == set(new)
    return base, new


def _short(k: Key) -> Key:
    return (k[0][:8], k[1], k[2])


def test_unaffected_questions_are_byte_identical_to_a8(replayed):
    base, new = replayed
    drift = sorted(_short(k) for k in base if _short(k) not in CHANGED and not _same(base[k], new[k]))
    assert drift == [], f"{len(drift)} questions drifted outside CHANGED: {drift[:10]}"
    assert sum(1 for k in base if _same(base[k], new[k])) == TOTAL - len(CHANGED)


def test_every_listed_change_is_real(replayed):
    base, new = replayed
    unchanged = sorted(_short(k) for k in base if _short(k) in CHANGED and _same(base[k], new[k]))
    assert unchanged == [], f"listed as changed but identical to a8: {unchanged}"


def test_target_5c2a85a4_s33_record_is_the_users_own_statement(replayed):
    base, new = replayed
    (k,) = [k for k in new if _short(k) == ("5c2a85a4", 33, "Q_001")]
    assert not _same(base[k], new[k])  # a8: Claire's father ("her father has alternate weekends") was the record
    record = new[k]["block"].splitlines()[2]
    assert "ESTABLISHED RECORD" in record and "I don't have a father" in record
    assert "her father" not in new[k]["block"]


def test_has_no_after_has_yes_yields_conflicting_line(replayed):
    base, new = replayed
    fixed = [("f398d596", 41), ("75345e85", 45), ("14e41890", 47), ("9dffc5de", 28), ("f47bb65a", 45), ("bb30a4e6", 52)]
    for short in fixed:
        (k,) = [k for k in new if _short(k)[:2] == short and k[2] == "Q_001"]
        assert not _same(base[k], new[k]), short  # a8 had no conflicting line for these (see memconflict_diff)
        assert "LATER CONFLICTING CLAIM" in new[k]["block"], short


def test_8cb3c9c6_s45_is_a_dataset_artifact_and_stays_identical(replayed):
    """Point_B of SC_004 (Has_Father: No) is labelled on session 45, but no user turn in that session - or any
    earlier one - states it. A ledger that reads only what the user said cannot and must not invent it."""
    base, new = replayed
    (k,) = [k for k in new if _short(k) == ("8cb3c9c6", 45, "Q_001")]
    assert _same(base[k], new[k])
    assert "LATER CONFLICTING CLAIM" not in new[k]["block"]


def test_replay_is_deterministic(replayed):
    _, new = replayed
    again = {(r["persona"], r["session"], r["question_id"]): _digest(r["block"]) for r in replay(str(_corpus()))}
    assert again == {k: r["block_sha256"] for k, r in new.items()}
