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


def test_birthdate_one_word_and_dob_statements_are_read():
    ledger = _ledger([
        (0, "My birthdate is 1980-03-22, in case the form asks."),
        (1, "Quick fix for the paperwork: my DOB is 1980-08-14."),
    ])
    resolved = ledger.resolve("birthdate")
    assert [r.status for r in resolved] == ["established", "conflicting"]  # same year, different date
    assert "1980-03-22" in resolved[0].statement.text
    q = "What is the user's birthdate?"
    block = ledger.render(ledger.route(q), detail=question_detail(q))
    assert "ESTABLISHED RECORD" in block and block.index("1980-03-22") < block.index("1980-08-14")


def test_birthdate_same_date_in_two_formats_is_consistent():
    ledger = _ledger([
        (0, "I was born on March 22, 1980."),
        (1, "For the record, I was born on 22 March 1980."),
    ])
    assert [r.status for r in ledger.resolve("birthdate")] == ["established", "consistent"]


def test_employment_question_gets_a_status_line_not_a_title_history():
    ledger = _ledger([
        (0, "Quick update: I'm now an intern at Northern Logistics."),
        (1, "Big news: I just started a new job as a director at Huaxin Consulting."),
    ])
    q = "Has the user's employment status stayed the same?"
    assert question_detail(q) == "employment"
    block = ledger.render(ledger.route(q), detail=question_detail(q))
    assert "before the latest change, employed (2024-01-01); after it, employed (2024-01-02); unchanged." in block
    assert "Huaxin Consulting" in block  # the statements themselves are still shown


def test_employment_question_reports_a_change_of_status():
    ledger = _ledger([
        (0, "I work as an analyst at a regional bank."),
        (1, "Quick update: I got laid off last week, so money is tight."),
    ])
    q = "Did the user's employment status remain the same?"
    block = ledger.render(ledger.route(q), detail=question_detail(q))
    assert "before the latest change, employed (2024-01-01); after it, unemployed (2024-01-02); changed." in block


def test_employment_kind_reads_the_last_state_named():
    from harpe.profile import _employment_kind
    assert _employment_kind("I got laid off in May but landed a new job last week.") == "employed"
    assert _employment_kind("I quit my job to start my own company.") == "self-employed or own business"
    assert _employment_kind("I'm job hunting while I keep my current role.") == "employed"
    assert _employment_kind("I'm balancing job hunting with some freelance work.") is None
    assert _employment_kind("Any tips on job hunting?") is None


def test_other_career_questions_get_no_employment_line():
    ledger = _ledger([(0, "I work as an analyst at a regional bank.")])
    q = "What is the user's job title?"
    assert question_detail(q) != "employment"
    assert "Employment status implied" not in ledger.render(ledger.route(q), detail=question_detail(q))
