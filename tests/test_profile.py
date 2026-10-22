"""ProfileLedger: the user's own statements about an attribute, resolved by whether the attribute can change."""

from harpe import ProfileLedger, question_detail


def _ledger(turns):
    """turns: (session, text) or (session, text, speaker); one turn per entry, in order."""
    ledger = ProfileLedger()
    for i, turn in enumerate(turns):
        session, text = turn[0], turn[1]
        speaker = turn[2] if len(turn) > 2 else "user"
        ledger.add(f"t{i}", text, date=f"2024-01-{session + 1:02d}", order=(session, i), speaker=speaker)
    return ledger


def test_mutable_attribute_latest_session_is_current():
    ledger = _ledger([
        (0, "I live in Denver with my two cats."),
        (1, "Big news: I just moved to Seattle for the new job."),
    ])
    resolved = ledger.resolve("residence")
    assert [r.status for r in resolved] == ["superseded", "current"]
    assert "Seattle" in resolved[-1].statement.text
    block = ledger.render(ledger.route("Where does the user live now?"))
    assert "CURRENT" in block and "Seattle" in block
    assert block.index("Denver") < block.index("Seattle")  # oldest first


def test_immutable_attribute_first_statement_is_the_record():
    ledger = _ledger([
        (0, "I graduated from California State University, Long Beach, so I know the area."),
        (3, "Since I studied at MIT, I like to problem-solve on the spot."),
    ])
    resolved = ledger.resolve("university")
    assert resolved[0].status == "established"
    assert "Long Beach" in resolved[0].statement.text
    assert resolved[1].status == "conflicting"
    block = ledger.render(ledger.route("Which university did the user attend?"))
    assert "ESTABLISHED RECORD" in block and "LATER CONFLICTING CLAIM" in block


def test_third_party_statements_are_not_the_user():
    ledger = _ledger([
        (0, "My dad graduated from Stanford University back in the day."),
        (0, "A colleague who went to the University of Toronto gave me advice."),
    ])
    assert ledger.statements("university") == []


def test_assistant_turns_are_not_read():
    ledger = _ledger([(0, "You mentioned you live in Austin, right?", "assistant")])
    assert ledger.statements("residence") == []
    assert ledger.latest_order is not None  # the turn still advances the timeline


def test_gender_needs_a_value():
    ledger = _ledger([
        (0, "I'm a guy, and one of my close friends here is married."),
        (2, "Oh, by the way, I identify as female and use she/her pronouns."),
    ])
    resolved = ledger.resolve("gender")
    assert [r.status for r in resolved] == ["established", "conflicting"]


def test_routing():
    ledger = ProfileLedger()
    assert ledger.route("Which university did the user attend?") == ["university"]
    assert "residence" in ledger.route("Did the user's residence change recently?")
    # conditional / preference questions are left to retrieval
    assert ledger.route("If the user is traveling, what should they pack?") == []
    assert question_detail("When was the user's mother born?") == "date"


def test_render_is_empty_without_statements():
    ledger = _ledger([(0, "Can you help me plan a budget?")])
    assert ledger.render(ledger.route("Where does the user live?")) == ""
    assert ledger.render([]) == ""


def test_other_peoples_birth_dates_are_not_the_users():
    ledger = _ledger([
        (0, "Just so you know, I have two kids now: Maya (born 2014-02-04) and Jackson (born 2017-03-25)."),
        (1, "I'm born 1988-05-14, so my routines have shifted."),
        (2, "Big update: I'm now dating Helen Wilson (born 1990-06-27)."),
        (3, "Oh, by the way, my birthday is May 14, 1990."),
    ])
    resolved = ledger.resolve("birthdate")
    assert [r.statement.text[:12] for r in resolved] == ["I'm born 198", "Oh, by the w"]
    assert [r.status for r in resolved] == ["established", "conflicting"]


def test_change_keeps_the_previous_state_over_later_questions():
    from harpe.profile import ProfileStatement, ResolvedStatement, _select

    def r(session, i, text, status):
        return ResolvedStatement(ProfileStatement("career", text, f"t{i}", None, (session, i), False, frozenset()), status)

    resolved = [r(0, 0, "Quick update: I'm now an intern at Northern Logistics.", "superseded")]
    resolved += [r(1, i, f"How do I plan my income question {i}?", "superseded") for i in range(1, 6)]
    resolved += [r(2, 6, "Big news: I just started as a director at Huaxin Consulting.", "current")]
    resolved += [r(2, i, f"What should I ask about benefits {i}?", "current") for i in range(7, 12)]
    shown = _select(resolved, True, 6)
    texts = [x.statement.text for x in shown]
    assert len(shown) == 6
    assert any("Northern Logistics" in t for t in texts) and any("Huaxin Consulting" in t for t in texts)
    assert [x.statement.order for x in shown] == sorted(x.statement.order for x in shown)
