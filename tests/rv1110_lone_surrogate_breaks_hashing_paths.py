"""severity: low
finding: text holding a lone surrogate (half of an escaped emoji cut by truncation) raises UnicodeEncodeError from DynamicToolCompressor.maybe_compress and from synthesize when the item is pinned; the same item unpinned is accepted (present at 723ff5b too, not introduced by PR #16)

Every hashing path encodes with strict UTF-8: the DTOC ref and byte size (src/harpe/dtoc.py:264, :277),
`pinned_policy_digest` (src/harpe/engine.py:109) and `dropped_ids_sha256` (src/harpe/two_pass.py:65). `json.loads`
turns a truncated `"\\ud83d\\ude00"` into a lone high surrogate, a normal Python str, and DTOC exists for large and
truncated tool output. The result is that the call meant to shrink a tool output raises instead, and that whether
`synthesize` accepts an item depends on whether it is pinned.

Correct behaviour asserted here: both calls accept any str (for example by hashing with
`errors="surrogatepass"`), and `restore` returns the output unchanged.
"""
import json

from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.dtoc import DynamicToolCompressor

HALF_EMOJI = json.loads('"status: done \\ud83d"')  # "\ud83d\ude00" cut after its first escape


def test_control_whole_emoji_compresses_and_restores():
    output = ("row of output\n" * 200) + json.loads('"status: done \\ud83d\\ude00"')
    comp = DynamicToolCompressor(min_compress_chars=100)
    _, artifact = comp.maybe_compress("terminal", output)
    assert artifact is not None and comp.restore(artifact.ref_id) == output


def test_control_unpinned_item_with_half_emoji_is_accepted():
    report = ContextBudgetEngine().synthesize([ContextItem(id="a", content=HALF_EMOJI)], TokenBudget(total_budget=1000))
    assert [it.id for it in report.included_items] == ["a"]


def test_truncated_tool_output_with_half_emoji_compresses_and_restores():
    output = ("row of output\n" * 200) + HALF_EMOJI
    comp = DynamicToolCompressor(min_compress_chars=100)
    placeholder, artifact = comp.maybe_compress("terminal", output)
    assert artifact is not None
    assert comp.restore(artifact.ref_id) == output
    assert DynamicToolCompressor.parse_placeholder(placeholder)["ref"] == artifact.ref_id


def test_pinned_item_with_half_emoji_is_accepted_like_an_unpinned_one():
    items = [ContextItem(id="a", content=HALF_EMOJI, pinned=True)]
    report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=1000))
    assert [it.id for it in report.included_items] == ["a"]
    assert len(report.pinned_policy_digest) == 64
