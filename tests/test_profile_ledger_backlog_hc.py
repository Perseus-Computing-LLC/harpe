"""ProfileLedger backlog items 5 and 6 (harpe 0.1.0a9 review).

5. A kinship noun owned by a third party ("her father", "Claire's father", "its mother") is about that third party,
   regardless of an "I" or "me" earlier in the sentence. It must not be read as the user's own parent or sibling, and
   a sentence whose subject is that relative must not be read as a statement about the user for any attribute.
6. An explicit "I don't have a ..." after a statement that implied the relative exists must be rendered as a later
   conflicting claim next to the established record, not as a restatement and not silently dropped.
"""

from harpe import ProfileLedger
from harpe.profile import _about_someone_else


def _ledger(turns):
    ledger = ProfileLedger()
    for i, (unit_id, text, date) in enumerate(turns):
        ledger.add(unit_id, text, date=date, order=(i + 1, 0))
    return ledger


# -- item 5: third-person possessive wins ---------------------------------------------------------------------------

def _start(sentence, noun):
    return sentence.index(noun)


def test_her_father_is_not_the_users_father():
    text = "Claire lives with me these days, and her father is visiting next week."
    assert _about_someone_else(text, _start(text, "father")) is True
    ledger = _ledger([("u1", text, "2024-03-01")])
    assert ledger.statements("father") == []
    assert ledger.render(["father"]) == ""


def test_named_possessive_kin_is_someone_else_despite_earlier_i():
    for text, noun in [
        ("I think Claire's sister lives in Boston.", "sister"),
        ("I spent the day with Claire, and Claire's father is visiting next week.", "father"),
        ("I was with Dan and his mother is visiting.", "mother"),
        ("I met the couple and their daughter starts school soon.", "daughter"),
        ("The company sent me a card and its founder's wife signed it.", "wife"),
    ]:
        assert _about_someone_else(text, _start(text, noun)) is True, text


def test_named_possessive_kin_is_not_recorded_for_any_attribute():
    ledger = _ledger([
        ("u1", "I spent the day with Claire, and Claire's father is visiting next week.", "2024-03-01"),
        ("u2", "I visited Claire and Claire's father lives in Boston now.", "2024-03-02"),
        ("u3", "I talked to Dan; its mother works as a nurse at the clinic.", "2024-03-03"),
    ])
    for attribute in ("father", "mother", "residence", "career"):
        assert ledger.statements(attribute) == [], attribute


def test_my_kin_is_still_the_users_own():
    text = "I visited Claire, and my father is visiting next week."
    assert _about_someone_else(text, _start(text, "father")) is False
    ledger = _ledger([("u1", text, "2024-03-01")])
    assert [s.attribute for s in ledger.statements("father")] == ["father"]


# -- item 6: a later has:no after an inferred has:yes is a conflicting claim ---------------------------------------

def _assert_conflict_rendered(ledger, attribute, first, later):
    resolved = ledger.resolve(attribute)
    assert [r.status for r in resolved] == ["established", "conflicting"], [(r.status, r.statement.text) for r in resolved]
    block = ledger.render([attribute])
    lines = block.splitlines()
    record = [line for line in lines if "ESTABLISHED RECORD" in line]
    conflict = [line for line in lines if "LATER CONFLICTING CLAIM" in line]
    assert len(record) == 1 and len(conflict) == 1, block
    assert first[1] in record[0] and first[0] in record[0], block
    assert later[1] in conflict[0] and later[0] in conflict[0], block
    assert block.index(record[0]) < block.index(conflict[0])


def test_explicit_no_father_after_inferred_father_is_a_conflicting_claim():
    first = ("2024-01-01", "I take my dad to the park every morning.")
    later = ("2024-02-01", "I don't have a dad.")
    ledger = _ledger([("u1", first[1], first[0]), ("u2", later[1], later[0])])
    assert "has:yes" in ledger.statements("father")[0].values
    assert "has:no" in ledger.statements("father")[1].values
    _assert_conflict_rendered(ledger, "father", first, later)


def test_explicit_no_sister_after_inferred_sister_is_a_conflicting_claim():
    first = ("2024-01-01", "I take my sister to the park every morning.")
    later = ("2024-02-01", "I don't have a sister.")
    ledger = _ledger([("u1", first[1], first[0]), ("u2", later[1], later[0])])
    assert "has:yes" in ledger.statements("siblings")[0].values
    values = ledger.statements("siblings")[1].values
    assert "has:no" in values and "has:yes" not in values, values
    assert not any(v.startswith("count:") and v != "count:0" for v in values), values
    _assert_conflict_rendered(ledger, "siblings", first, later)


def test_explicit_no_brother_or_sister_after_inferred_brother_is_a_conflicting_claim():
    first = ("2024-01-01", "My brother and I play chess on Sundays.")
    later = ("2024-02-01", "Actually, I do not have a brother or a sister.")
    ledger = _ledger([("u1", first[1], first[0]), ("u2", later[1], later[0])])
    assert "has:no" in ledger.statements("siblings")[1].values
    _assert_conflict_rendered(ledger, "siblings", first, later)


def test_explicit_no_mother_after_inferred_mother_is_a_conflicting_claim():
    first = ("2024-01-01", "I call my mom every Sunday.")
    later = ("2024-02-01", "I never had a mother.")
    ledger = _ledger([("u1", first[1], first[0]), ("u2", later[1], later[0])])
    _assert_conflict_rendered(ledger, "mother", first, later)
