"""Pass 2 of the two-pass context build (telarch P14.3, master roadmap 17.2) and its ``harpe.context_built`` receipt.

Pass 1 places every kept row as a description. Pass 2 expands a row's body when the caller requests it (the
turn's query or the agent named the row), in request order, while the remaining budget allows; a request that does
not fit is refused and the row stays a description. Pass 2 only swaps a row's description for its body: it never
removes an item and never touches tier 0 (pinned, priority-0 and invariant items), so an expansion that would need
a tier-0 item's tokens is refused and the pinned set and its ``pinned_policy_digest`` are those of Pass 1.

Each build emits at most one ``harpe.context_built`` receipt, ints and strings only, written to the Argus ledger
whose path the caller passes. There is no default path: a caller that does not name a ledger cannot write one.
Writing the receipt seals the build: a later ``request`` raises, so the context the build hands out is always the
one its receipt describes. The build is sealed before the ledger write starts, not after it returns. An ``emit``
whose write raises may have left its receipt in the ledger (Argus can fail after the record's bytes are written),
so the build stays sealed and refuses further requests and a second ``emit``: a crash cannot leave two receipts
for one build, or a context that differs from a receipt already on disk. The build keeps its own copies of the
items it is given and hands out copies. A copy
owns every field the receipt accounts for and the dicts, lists, tuples and sets inside ``metadata``; any other
object held there (a lock, a client, an open handle) is shared, never cloned.

A build continues a Pass 1 made for one destination, ``"remote"`` unless the caller says ``"local"``, as
:meth:`ContextBudgetEngine.synthesize` does. For ``"remote"`` no confidential item may enter the context: a report
that includes one, or a description or body marked confidential, is refused when the build is made.
``argus_ledger`` is imported only when a receipt is written (``pip install harpe[audit]``).
"""
from __future__ import annotations

import errno
import hashlib
import os
import stat
import threading
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .engine import ContextBudgetEngine, pinned_policy_digest
from .models import ContextItem, SynthesisReport, _detached_copy, _utf8

__all__ = [
    "CONTEXT_BUILT_EVENT",
    "CONTEXT_BUILT_FIELDS",
    "TwoPassBuild",
    "append_context_built",
    "dropped_ids_sha256",
    "pinned_policy_digest",
]

CONTEXT_BUILT_EVENT = "harpe.context_built"
CONTEXT_BUILT_FIELDS: Tuple[str, ...] = (
    "pinned_policy_digest",
    "rows_pass1",
    "rows_expanded",
    "expansions_refused",
    "budget_tokens",
    "used_tokens",
    "dropped_ids_sha256",
)

LedgerPath = Union[str, "os.PathLike[str]"]


def _dropped_id_record(item_id: Any) -> str:
    text = str(item_id)
    if not text:
        return "\\"  # a lone backslash, which no escaped id can be
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def dropped_ids_sha256(ids: Iterable[str]) -> str:
    """sha256 (hex) of the ids sorted and joined by "\\n"; the hash of the empty string when there are none.

    Within an id a backslash is written ``\\\\`` and a newline ``\\n``, and an empty id is a lone ``\\``, so one id
    cannot pass for several and the hash names exactly one set of ids. Non-empty ids without a backslash or a
    newline hash as they did before these rules."""
    return hashlib.sha256(_utf8("\n".join(sorted(_dropped_id_record(i) for i in ids)))).hexdigest()


def _check_payload(payload: Mapping[str, Any]) -> Dict[str, Union[int, str]]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, not {type(payload).__name__}")
    missing = [k for k in CONTEXT_BUILT_FIELDS if k not in payload]
    extra = sorted(str(k) for k in payload if k not in CONTEXT_BUILT_FIELDS)
    if missing or extra:
        raise ValueError(f"harpe.context_built fields: missing {missing}, unexpected {extra}")
    for key in CONTEXT_BUILT_FIELDS:
        value = payload[key]
        # Exact types: bool is an int subclass and floats break the ledger's canonical hashing.
        if type(value) not in (int, str):
            raise TypeError(f"harpe.context_built field {key!r} must be an int or a str, not {type(value).__name__}")
    return {key: payload[key] for key in CONTEXT_BUILT_FIELDS}


def _check_ledger_path(ledger_path: Any) -> str:
    if ledger_path is None or not isinstance(ledger_path, (str, os.PathLike)):
        raise TypeError(f"ledger_path must be a str or os.PathLike, not {type(ledger_path).__name__}")
    path = os.fspath(ledger_path)
    if not isinstance(path, str) or not path:
        raise ValueError("ledger_path must name a file")
    return path


# The gates Argus applies inside ``append`` before it opens the ledger (argus_ledger.canonical): an actor is
# printable ASCII without whitespace, quotes or backslashes, at most 256 bytes; a payload int lies within
# +/-(2**53 - 1). Applied here as well, so that a receipt Argus would refuse is refused before the build is sealed.
_MAX_ACTOR_BYTES = 256
_MAX_SAFE_INTEGER = 2 ** 53 - 1


def _check_actor(actor: Any) -> str:
    if not isinstance(actor, str) or not actor:
        raise ValueError("actor must be a non-empty str")
    if (len(actor) > _MAX_ACTOR_BYTES or any(not ("\x21" <= ch <= "\x7e") for ch in actor)
            or '"' in actor or "\\" in actor):
        raise ValueError(f"actor {actor!r} must be printable ASCII without whitespace, quotes or backslashes "
                         f"(1-{_MAX_ACTOR_BYTES} bytes), as the Argus ledger requires")
    return actor


def _check_payload_range(record: Mapping[str, Union[int, str]]) -> None:
    for key, value in record.items():
        if type(value) is int and abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError(f"harpe.context_built field {key!r} is {value}; the Argus ledger takes integers "
                             f"within +/-{_MAX_SAFE_INTEGER}")


def _probe_ledger_path(path: str) -> None:
    """Refuse a ``path`` whose first ledger syscall would fail, without creating or writing anything.

    The file is opened the way the Argus writer opens it (read-write, append, no create) and closed again. A path
    that does not exist yet is accepted when its nearest existing ancestor is a directory this process can write
    in, since Argus then creates the missing directories and the file. A directory, or any other file that is not a
    regular file, cannot hold a receipt and is refused."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CLOEXEC)
    except FileNotFoundError:
        ancestor = os.path.dirname(os.path.abspath(path))
        while not os.path.lexists(ancestor):
            ancestor = os.path.dirname(ancestor)
        if not os.path.isdir(ancestor):
            raise NotADirectoryError(errno.ENOTDIR, "the ledger's nearest existing ancestor is not a directory",
                                     ancestor) from None
        if not os.access(ancestor, os.W_OK | os.X_OK):
            raise PermissionError(errno.EACCES, "the ledger file cannot be created in", ancestor) from None
        return
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "the ledger path is not a regular file", path)
    finally:
        os.close(fd)


def _prepared_context_built(payload: Mapping[str, Any], *, ledger_path: LedgerPath,
                            actor: str) -> Callable[[], int]:
    """Check everything that can be checked without writing to the ledger, and return the one step that writes.

    Every refusal raises here, before any byte is written: the payload's fields and types, the int range and the
    actor grammar Argus enforces (with Argus's own validators too, when the installed ``argus_ledger`` exports
    them), a ``ledger_path`` that cannot be opened or created as a regular file, a missing ``argus_ledger``. So a
    caller can tell a receipt that was never attempted from one whose write began. A failure that appears only
    once the returned step runs (the path changed in between, a lock, a read or write error) is not one of these:
    that write may have begun."""
    record = _check_payload(payload)
    _check_payload_range(record)
    path = _check_ledger_path(ledger_path)
    _check_actor(actor)
    try:
        from argus_ledger import AppendOnlyLedger
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError("writing a harpe.context_built receipt needs argus-ledger (pip install harpe[audit])") from exc
    try:
        from argus_ledger import validate_identifiers, validate_payload
    except ImportError:  # pragma: no cover - an argus-ledger that predates the exported gates
        pass
    else:
        validate_identifiers(CONTEXT_BUILT_EVENT, actor)
        validate_payload(dict(record))
    ledger = AppendOnlyLedger(path)
    _probe_ledger_path(path)

    def write() -> int:
        event = ledger.append(CONTEXT_BUILT_EVENT, actor, record)
        return int(event.sequence)

    return write


def append_context_built(payload: Mapping[str, Any], *, ledger_path: LedgerPath, actor: str = "harpe") -> int:
    """Append one ``harpe.context_built`` event with ``payload`` to the Argus ledger at ``ledger_path``; return
    its sequence. The payload must carry exactly ``CONTEXT_BUILT_FIELDS``, each an int or a str; nothing is
    written when it does not.

    When the append itself raises, the record may already be in the ledger: Argus can fail after the record's
    bytes are written (a failed fsync whose rollback also fails, a ``close`` that reports EIO after the fsync).
    Do not take an exception from the append to mean that no receipt exists."""
    return _prepared_context_built(payload, ledger_path=ledger_path, actor=actor)()


class TwoPassBuild:
    """Pass 2 over a Pass-1 ``SynthesisReport``.

    The report's unpinned included items are the Pass-1 rows. Give exactly one mapping, keyed by those rows' ids:

    * ``bodies``: the report selected on descriptions (each row is its description); the mapping holds the bodies;
    * ``descriptions``: the report selected on bodies (each row is its body); the mapping holds the descriptions.

    Either way each row starts as its description. ``budget_tokens`` is the ceiling Pass 2 spends against: tier-0
    tokens plus every row's current form. Items without a token count (``token_count <= 0``) are counted with
    ``count_tokens``; without it they are refused.

    ``destination`` is the one Pass 1 was synthesized for (default ``"remote"``, as in ``synthesize``). For
    ``"remote"`` the report must include no confidential item and no description or body may be confidential:
    either raises ``ValueError``, so Pass 2 cannot send what Pass 1 withheld. A report that repeats an id, across
    its included, pruned and withheld items, raises ``ValueError`` too: rows are keyed by id, and a repeat would
    hand out a context the receipt does not describe.
    """

    def __init__(
        self,
        report: SynthesisReport,
        *,
        descriptions: Optional[Mapping[str, ContextItem]] = None,
        bodies: Optional[Mapping[str, ContextItem]] = None,
        budget_tokens: int,
        count_tokens: Optional[Callable[[str], int]] = None,
        destination: str = "remote",
    ) -> None:
        if (descriptions is None) == (bodies is None):
            raise ValueError("give exactly one of descriptions= or bodies=")
        if destination not in ("remote", "local"):
            raise ValueError("destination must be 'remote' or 'local'")
        remote = destination == "remote"
        if type(budget_tokens) is not int:
            raise TypeError(f"budget_tokens must be an int, not {type(budget_tokens).__name__}")
        if budget_tokens < 0:
            raise ValueError("budget_tokens must be >= 0")
        self._count_tokens = count_tokens
        self._budget = budget_tokens

        rows_are_bodies = descriptions is not None
        given: Mapping[str, ContextItem] = descriptions if rows_are_bodies else bodies  # type: ignore[assignment]
        # Rows, tier-0 items and dropped ids are all keyed by id, so an id the report repeats would be handed out
        # twice but counted once, or named as dropped while it is in the context: the receipt would describe
        # another context than items(). Refuse the report rather than pick one occurrence.
        ContextBudgetEngine._refuse_duplicate_ids(
            [*report.included_items, *report.pruned_items, *report.withheld_items], " in the report"
        )
        self._order: List[Tuple[str, bool]] = []  # (id, is_tier0) in the report's included order
        self._tier0: Dict[str, ContextItem] = {}
        self._desc: Dict[str, ContextItem] = {}
        self._body: Dict[str, ContextItem] = {}
        for item in report.included_items:
            if remote and ContextBudgetEngine._withheld_from_remote(item):
                raise ValueError(f"the report includes confidential item {item.id!r}, which a remote destination "
                                 "withholds; pass destination='local' if Pass 1 was built for local")
            if ContextBudgetEngine._is_never_pruned(item):
                self._tier0[item.id] = self._counted(item)
                self._order.append((item.id, True))
            else:
                (self._body if rows_are_bodies else self._desc)[item.id] = self._counted(item)
                self._order.append((item.id, False))

        row_ids = set(self._body if rows_are_bodies else self._desc)
        missing = sorted(row_ids - set(given))
        extra = sorted(str(k) for k in set(given) - row_ids)
        if missing or extra:
            raise ValueError(f"rows must match the report's unpinned included items: missing {missing}, "
                             f"not a Pass-1 row {extra}")
        for row_id in sorted(row_ids):
            item = given[row_id]
            if isinstance(item, dict):
                item = ContextItem.model_validate(item)
            if not isinstance(item, ContextItem):
                raise TypeError(f"row {row_id!r}: expected a ContextItem, not {type(item).__name__}")
            if item.id != row_id:
                raise ValueError(f"row {row_id!r}: the item's id is {item.id!r}")
            if ContextBudgetEngine._is_never_pruned(item):
                raise ValueError(f"row {row_id!r}: a description or body cannot be a tier-0 item "
                                 "(pinned, priority 0 or invariant)")
            if remote and ContextBudgetEngine._withheld_from_remote(item):
                raise ValueError(f"row {row_id!r}: its {'description' if rows_are_bodies else 'body'} is "
                                 "confidential, and a remote destination withholds confidential items")
            (self._desc if rows_are_bodies else self._body)[row_id] = self._counted(item)

        self._dropped: Tuple[str, ...] = tuple(it.id for it in report.pruned_items) + tuple(
            it.id for it in report.withheld_items
        )
        self._expanded: Dict[str, None] = {}  # insertion-ordered set: expansion order
        self._refused = 0
        self._used = sum(it.token_count for it in self._tier0.values()) + sum(
            it.token_count for it in self._desc.values()
        )
        self._receipt_sequence: Optional[int] = None
        # Set before the ledger write starts and never cleared: once a write may have reached the ledger, the build
        # must neither change nor write again, whether or not the write returned a sequence.
        self._sealed = False
        self._lock = threading.Lock()  # a request cannot land between the receipt's payload and its write

    def _counted(self, item: ContextItem) -> ContextItem:
        """The build's own copy of ``item``, counted: later edits to the caller's object do not reach the build.
        The copy is :func:`harpe.models._detached_copy`, which does not clone the objects ``metadata`` holds, so
        an item carrying one that cannot be copied builds like any other."""
        if item.token_count > 0:
            return _detached_copy(item)
        if self._count_tokens is None:
            raise ValueError(f"item {item.id!r} has no token count; pass count_tokens=")
        tokens = self._count_tokens(item.content)
        if type(tokens) is not int or tokens < 0:
            raise ValueError(f"count_tokens returned {tokens!r} for item {item.id!r}")
        return _detached_copy(item, {"token_count": tokens})

    # ---- Pass 2 ----
    def request(self, item_id: str) -> bool:
        """Expand row ``item_id`` if its body fits in what the budget leaves; True when the row is (now) in full.

        A tier-0 item is always in full (True, nothing spent); an expanded row stays expanded (True); a request
        that does not fit is refused (False, counted in ``expansions_refused``) and the row stays a description.
        An id that is not in this build raises ``KeyError``. Once the receipt is written (:meth:`emit`) the build is
        sealed and every request raises ``RuntimeError``: the context may not change under its receipt."""
        with self._lock:
            if self._sealed:
                raise RuntimeError(f"{self._seal_reason()}; a request after emit() cannot change the context it "
                                   "describes")
            if item_id in self._tier0 or item_id in self._expanded:
                return True
            if item_id not in self._desc:
                raise KeyError(item_id)
            delta = self._body[item_id].token_count - self._desc[item_id].token_count
            if self._used + delta > self._budget:
                self._refused += 1
                return False
            self._used += delta
            self._expanded[item_id] = None
            return True

    def is_expanded(self, item_id: str) -> bool:
        return item_id in self._expanded

    @property
    def expanded_ids(self) -> Tuple[str, ...]:
        return tuple(self._expanded)

    @property
    def dropped_ids(self) -> Tuple[str, ...]:
        """Ids the Pass-1 report left out of the context: pruned, then withheld."""
        return self._dropped

    @property
    def rows_pass1(self) -> int:
        return len(self._desc)

    @property
    def rows_expanded(self) -> int:
        return len(self._expanded)

    @property
    def expansions_refused(self) -> int:
        return self._refused

    @property
    def budget_tokens(self) -> int:
        return self._budget

    @property
    def used_tokens(self) -> int:
        return self._used

    def items(self) -> List[ContextItem]:
        """The context in the report's included order: tier-0 items in full, each row as its body when expanded and
        as its description otherwise. The items are copies: editing them does not change the build."""
        out: List[ContextItem] = []
        for item_id, is_tier0 in self._order:
            if is_tier0:
                item = self._tier0[item_id]
            else:
                item = self._body[item_id] if item_id in self._expanded else self._desc[item_id]
            out.append(_detached_copy(item))
        return out

    # ---- Receipt ----
    def receipt_payload(self) -> Dict[str, Union[int, str]]:
        """The ``harpe.context_built`` payload: ``pinned_policy_digest`` is recomputed over this build's tier-0
        items, so it equals the Pass-1 report's digest exactly when Pass 2 left the pinned set alone."""
        return {
            "pinned_policy_digest": pinned_policy_digest(list(self._tier0.values())),
            "rows_pass1": self.rows_pass1,
            "rows_expanded": self.rows_expanded,
            "expansions_refused": self.expansions_refused,
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "dropped_ids_sha256": dropped_ids_sha256(self._dropped),
        }

    def _seal_reason(self) -> str:
        if self._receipt_sequence is not None:
            return f"this build's receipt is already at sequence {self._receipt_sequence}"
        return ("an earlier emit() of this build failed while writing its receipt, which may be in the ledger; "
                "the build stays sealed (make a new build to write another receipt)")

    def emit(self, *, ledger_path: LedgerPath, actor: str = "harpe") -> int:
        """Write this build's one ``harpe.context_built`` receipt to ``ledger_path``; return its sequence. A
        second call raises ``RuntimeError``. The receipt seals the build: later requests raise.

        A refusal that writes nothing leaves the build open, so the call can be repeated: a payload or ``actor``
        the Argus ledger would refuse (an actor with whitespace, an int beyond +/-(2**53 - 1)), a ``ledger_path``
        that cannot be opened or created as a regular file (a directory, a missing parent that is a file, no write
        permission), ``argus_ledger`` not installed. All of these are checked before the build is sealed. The
        build is sealed before the ledger write starts: when the write raises, the receipt may already be in the
        ledger, so the build stays sealed with no
        ``receipt_sequence``, and later requests and emits raise ``RuntimeError`` rather than change the context
        under that receipt or write a second one."""
        with self._lock:
            if self._sealed:
                raise RuntimeError(self._seal_reason())
            write = _prepared_context_built(self.receipt_payload(), ledger_path=ledger_path, actor=actor)
            self._sealed = True  # before the first byte: a write that raises may still have left the receipt
            self._receipt_sequence = write()
            return self._receipt_sequence

    @property
    def sealed(self) -> bool:
        """True once :meth:`emit` has started writing the receipt, whether or not the write returned."""
        return self._sealed

    @property
    def receipt_sequence(self) -> Optional[int]:
        """The receipt's ledger sequence; None before :meth:`emit`, and after an ``emit`` whose write raised."""
        return self._receipt_sequence
