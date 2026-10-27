"""Diff two MemConflict ledger replays (tests.memconflict_replay output) question by question.

    python -m tests.memconflict_diff BASE.jsonl NEW.jsonl [--show N]

Prints the count of byte-identical questions, lists every question whose block or routed attributes changed,
and for each changed question the added / removed ledger lines. Exit 0 always; the test module asserts.
"""
from __future__ import annotations

import difflib
import json
import sys
from typing import Dict, List, Tuple

Key = Tuple[str, int, str]


def load(path: str) -> Dict[Key, dict]:
    out: Dict[Key, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(r["persona"], r["session"], r["question_id"])] = r
    return out


def changed(base: Dict[Key, dict], new: Dict[Key, dict]) -> List[Key]:
    if set(base) != set(new):
        raise SystemExit(f"question sets differ: {len(base)} vs {len(new)}")
    return [k for k in base if base[k]["block"] != new[k]["block"] or base[k]["attributes"] != new[k]["attributes"]]


def line_diff(a: str, b: str) -> List[str]:
    return [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def main(argv: List[str]) -> int:
    show = 50
    if "--show" in argv:
        i = argv.index("--show"); show = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    base, new = load(argv[1]), load(argv[2])
    diff = changed(base, new)
    print(f"{len(base)} questions; {len(base) - len(diff)} byte-identical; {len(diff)} changed")
    for k in diff[:show]:
        b, n = base[k], new[k]
        print(f"\n== {k[0][:8]} s{k[1]} {k[2]} [{n.get('conflict_type')}] {n['question']}")
        if b["attributes"] != n["attributes"]:
            print(f"   attributes {b['attributes']} -> {n['attributes']}")
        for l in line_diff(b["block"], n["block"]):
            print("  ", l[:220])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
