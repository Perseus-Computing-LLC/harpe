"""Offline ProfileLedger replay over the MemConflict corpus: the ledger block each question would see.

Reproduces exactly what the MemConflict harness does with --profile (telarch_evals.memory.TelarchMemory, wheel
telarch_evals-0.1.0 of the Round 5 run): per persona, ingest each session in order (one unit per dialogue round,
one ledger.add per turn), and after each session's ingest route + render the block for each of its questions.

No model, no network, no paid call. Used by test_memconflict_replay.py to diff a harpe build against the blocks
recorded in a banked run (Retrieved_Memories[0] of every question whose retrieval carried profile_attributes).

    python -m tests.memconflict_replay DATASET.jsonl OUT.jsonl   # writes one row per question

Dataset: MemConflict Step4_4.jsonl (TaoZhen1110/MemConflict @ ec51d5d, 30 personas, 3,750 questions,
SHA-256 8ef9ec8589eccb86f63ab3a819a9180217405351a8d5846866721ea74babe092).
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterator, List

from harpe import ProfileLedger, question_detail


def _turn_order(key: str) -> int:
    try:
        return int(str(key).split("_")[-1])
    except Exception:
        return 10 ** 9


def dialogue_list(session_dialogue: Any) -> List[Dict[str, str]]:
    """eval_common.Build_Session_Dialogue_List: the session's turns in order, user/assistant only."""
    if not isinstance(session_dialogue, dict):
        return []
    out: List[Dict[str, str]] = []
    for key in sorted(session_dialogue.keys(), key=_turn_order):
        value = session_dialogue.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and content not in (None, ""):
                out.append({"role": role, "content": str(content)})
    return out


def rounds_of(dialogue: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """eval_telarch.rounds_of: one round per user turn plus the assistant turns that follow it."""
    rounds: List[List[Dict[str, str]]] = []
    for turn in dialogue:
        if turn["role"] == "user" or not rounds:
            rounds.append([turn])
        else:
            rounds[-1].append(turn)
    return rounds


def replay_persona(persona: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield {persona, session, question_id, question, attributes, block} for every question, in harness order."""
    ledger = ProfileLedger()
    pid = str(persona.get("ID", ""))
    for si, session in enumerate(persona.get("Full_Session_Chain", [])):
        date = str(session.get("Date", "") or "")
        # TelarchMemory._feed: units sorted by order, ledger.add per turn with uid "{si}:{ri}#{ti}"
        for ri, turns in enumerate(rounds_of(dialogue_list(session.get("Session_Dialogue", {})))):
            for ti, turn in enumerate(turns):
                ledger.add(f"{si}:{ri}#{ti}", turn["content"], date=date or None, order=(si, ri, ti),
                           speaker=turn["role"])
        for q in session.get("Session_Questions", []) or []:
            question = str(q.get("question", ""))
            attributes = ledger.route(question)
            block = ledger.render(attributes, detail=question_detail(question)) if attributes else ""
            yield {"persona": pid, "session": int(session.get("Session_ID", si)), "question_id": q.get("question_id"),
                   "conflict_type": q.get("conflict_type"), "question": question,
                   "attributes": list(attributes), "block": block}


def replay(dataset_path: str) -> Iterator[Dict[str, Any]]:
    with open(dataset_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield from replay_persona(json.loads(line))


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    n = 0
    with open(argv[2], "w", encoding="utf-8") as out:
        for row in replay(argv[1]):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} questions replayed -> {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
