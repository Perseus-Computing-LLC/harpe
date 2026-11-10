"""severity: high
finding: DTOC refs are a 48-bit truncation of sha256 and the store overwrites on a ref match without comparing payloads, so two different tool outputs share a ref and restore() of the first returns the second

CHANGELOG.md ([Unreleased], Fixed, "Digests and handles") claims that with the tool name escaped "two different
artifacts cannot share a ref and overwrite each other's stored output". The fix escaped ``\\`` and ``:`` in the
tool name (src/harpe/dtoc.py:223-226), which closes only the structural collision. The ref itself is
``"dtoc_" + sha256(f"{safe_tool}:{output}").hexdigest()[:12]`` (dtoc.py:227-228), 48 bits, and ``maybe_compress``
stores the artifact with ``self._store[ref_id] = artifact`` (dtoc.py:257). ``LRUArtifactStore.__setitem__``
(dtoc.py:96-106) replaces whatever is under that key without comparing the payload, the tool or the sidecar. A
pair of different outputs with the same 48-bit prefix needs only a birthday search of about 2**24 hashes: the pair
below was found in about 20 seconds of pure Python, with no weakness in SHA-256. Anyone who controls the text of
two tool outputs (two web pages, two files) can make one.

The consequences:

* the first artifact's stored output is lost, and ``restore(first.ref_id)`` (dtoc.py:294-299), the expansion its
  placeholder tells the model to request, returns the second artifact's output;
* the two artifacts' context items share the id ``item_<ref>`` (dtoc.py:285-286);
* the overwrite ignores the sidecar. Here the first artifact is an ordinary public page (``confidential`` unset,
  so its placeholder and item go to a remote model), the second is marked confidential, and the public
  placeholder's ref now restores the confidential payload.

The test accepts any fix: refs long enough not to collide, a store that refuses or disambiguates a ref already
holding a different output, or ``maybe_compress`` refusing the colliding artifact.
"""
from __future__ import annotations

from harpe import DynamicToolCompressor

TOOL = "http_get"

_PUBLIC_BASE = (
    "GET https://status.example.com/ -> 200\n"
    "All systems operational.\n"
    + "".join(f"component {i}: ok\n" for i in range(12))
    + "request-id: "
)
_PAYROLL_BASE = (
    "GET https://intranet.example.com/payroll/2026-10 -> 200\n"
    "employee,salary\n"
    "alice,182000\n"
    "bob,176500\n"
    + "".join(f"row {i}: restricted\n" for i in range(10))
    + "request-id: "
)

# Found by a distinguished-point birthday search over the 12-hex request id (about 4e7 sha256 calls): with the
# tool "http_get", both outputs hash to the ref prefix 8f0207e1b5f1.
PUBLIC_PAGE = _PUBLIC_BASE + "072814fb983c"
PAYROLL_PAGE = _PAYROLL_BASE + "b1e525878ae9"


def _compressor() -> DynamicToolCompressor:
    return DynamicToolCompressor(min_compress_chars=100)


def test_control_two_ordinary_outputs_restore_to_their_own_text():
    comp = _compressor()
    public = _PUBLIC_BASE + "000000000001"
    payroll = _PAYROLL_BASE + "000000000002"
    _, a = comp.maybe_compress(TOOL, public)
    _, b = comp.maybe_compress(TOOL, payroll, confidential=True)
    assert a is not None and b is not None and a.ref_id != b.ref_id
    assert comp.restore(a.ref_id) == public
    assert comp.restore(b.ref_id) == payroll
    assert comp.to_context_item(a).sensitivity == "normal"
    assert comp.to_context_item(b).sensitivity == "confidential"


def test_a_second_different_output_does_not_replace_the_first_artifacts_stored_output():
    comp = _compressor()
    _, public = comp.maybe_compress(TOOL, PUBLIC_PAGE)
    assert public is not None and public.confidential is False
    try:
        _, payroll = comp.maybe_compress(TOOL, PAYROLL_PAGE, confidential=True)
    except (ValueError, RuntimeError, KeyError):
        payroll = None  # refusing the colliding artifact is a correct fix
    restored = comp.restore(public.ref_id)
    assert restored == PUBLIC_PAGE, (
        f"restore({public.ref_id!r}), the public page's ref, returned another artifact's output: "
        f"{(restored or '')[:60]!r}"
    )
    if payroll is not None:
        assert comp.restore(payroll.ref_id) == PAYROLL_PAGE
