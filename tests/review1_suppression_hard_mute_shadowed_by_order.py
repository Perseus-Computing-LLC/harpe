"""severity: medium
finding: SuppressionController applies the first matching rule in registration order, so a hard_mute rule added after any soft rule on the same item never mutes it

SuppressionRule.hard_mute is documented as "If True, completely excludes matching items from context synthesis"
(src/harpe/suppression.py:40-43). SuppressionController.evaluate_item (suppression.py:127-138) walks the active
rules in insertion order and returns on the first match, and filter_and_damp (suppression.py:140-168) hard-mutes
only when that first factor is 0.0. Whether an item is muted therefore depends on the order in which rules were
registered: a broad soft rule registered earlier (a category damped for being noisy) shadows a later hard mute of
one specific item (the user asked for it to be dropped). The item is retained, only tagged with a
``suppression_discount`` that no selection code reads (grep finds the key only in suppression.py), so the muted
content still reaches synthesize() at full utility.
"""
from __future__ import annotations

from harpe import ContextItem, SuppressionController, SuppressionScope


def _items():
    return [
        ContextItem(id="old_token_note", content="The old deploy token was tok-0000.", metadata={"category": "ops"},
                    token_count=9),
        ContextItem(id="ops_status", content="The build is green.", metadata={"category": "ops"}, token_count=5),
    ]


def _soft(c: SuppressionController):
    c.add_rule(SuppressionScope.CATEGORY, "ops", "ops chatter is noisy today", discount_factor=0.5)


def _hard(c: SuppressionController):
    c.add_rule(SuppressionScope.EXACT_ID, "old_token_note", "user asked to drop this note", hard_mute=True)


def _split(c: SuppressionController):
    retained, muted = c.filter_and_damp(_items())
    return [i.id for i in retained], [i.id for i in muted]


def test_control_hard_mute_alone_excludes_the_item():
    c = SuppressionController()
    _hard(c)
    assert _split(c) == (["ops_status"], ["old_token_note"])


def test_control_hard_mute_registered_before_the_soft_rule_excludes_the_item():
    c = SuppressionController()
    _hard(c)
    _soft(c)
    assert _split(c) == (["ops_status"], ["old_token_note"])


def test_a_hard_mute_excludes_the_item_even_when_a_soft_rule_was_registered_first():
    c = SuppressionController()
    _soft(c)
    _hard(c)
    retained, muted = _split(c)
    assert muted == ["old_token_note"], f"the hard-muted item was retained: retained={retained} muted={muted}"
    assert retained == ["ops_status"]
