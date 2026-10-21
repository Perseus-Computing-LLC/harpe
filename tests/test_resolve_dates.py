"""Relative dates resolved against the session date: "yesterday" said on 11 May 2023 gains [10 May 2023]."""

import pytest
from harpe import parse_anchor_date, resolve_relative_dates


@pytest.mark.parametrize(
    "anchor, iso",
    [
        ("1:56 pm on 8 May, 2023", "2023-05-08"),
        ("2023/05/20 (Sat) 02:21", "2023-05-20"),
        ("March-15-2024", "2024-03-15"),
        ("2024-03-15", "2024-03-15"),
        ("May 8, 2023", "2023-05-08"),
        ("", None),
        ("sometime last spring", None),
    ],
)
def test_anchor_formats(anchor, iso):
    parsed = parse_anchor_date(anchor)
    assert (parsed.isoformat() if parsed else None) == iso


@pytest.mark.parametrize(
    "text, anchor, expected",
    [
        ("I interviewed yesterday.", "11 May, 2023", "I interviewed yesterday [10 May 2023]."),
        ("We met last weekend!", "7 November, 2022", "We met last weekend [5-6 November 2022]!"),
        ("Bought it three days ago", "29 April, 2022", "Bought it three days ago [26 April 2022]"),
        ("I started last Friday", "13 March, 2023", "I started last Friday [10 March 2023]"),
        ("Two weeks ago I moved", "2023/05/20 (Sat) 02:21", "Two weeks ago [6 May 2023] I moved"),
        ("Launch is next week", "2023/03/15 (Wed) 10:00", "Launch is next week [week of 20 March 2023]"),
        ("Revenue dipped last month", "March-15-2024", "Revenue dipped last month [February 2024]"),
        ("the day before yesterday it rained", "8 May, 2023", "the day before yesterday [6 May 2023] it rained"),
        ("see you tomorrow", "31 December, 2023", "see you tomorrow [1 January 2024]"),
        ("last weekend", "1 March, 2023", "last weekend [25-26 February 2023]"),
        ("last year was hard", "8 May, 2023", "last year [2022] was hard"),
        ("we ship 10 days from now", "8 May, 2023", "we ship 10 days from now [18 May 2023]"),
        ("I lost 5 pounds in two weeks", "8 May, 2023", "I lost 5 pounds in two weeks"),
    ],
)
def test_relative_expressions_get_their_dates(text, anchor, expected):
    assert resolve_relative_dates(text, anchor) == expected


def test_text_without_an_anchor_or_relative_words_is_unchanged():
    assert resolve_relative_dates("I went yesterday", "") == "I went yesterday"
    assert resolve_relative_dates("I went yesterday", "no date here") == "I went yesterday"
    assert resolve_relative_dates("Paris in May 2023", "8 May, 2023") == "Paris in May 2023"


def test_resolving_twice_changes_nothing_more():
    once = resolve_relative_dates("yesterday and last Friday", "13 March, 2023")
    assert once == "yesterday [12 March 2023] and last Friday [10 March 2023]"
    assert resolve_relative_dates(once, "13 March, 2023") == once


def test_words_inside_other_words_are_left_alone():
    assert resolve_relative_dates("the yesterdays of old; todays", "8 May, 2023") == "the yesterdays of old; todays"
