//! StateLedger: pairing of a first-person claim with its later denial (or the reverse).
//!
//! Mirror of `src/harpe/state_ledger.py` with the same names and the same lexical rules, written without
//! dependencies so the crate stays dependency-free. The ledger reads only the dialogue units it is given (unit id,
//! speaker-tagged text, date) and the question or terms at query time.
//!
//! * `conflicts_for(question)` - the prompt path, gated by the question shape; empty unless the question asks
//!   about a first-person past action or state.
//! * `conflicts_for_entity(terms, window)` - the programmatic path: no gate, optional inclusive date window,
//!   threshold and limit.
//! * `timeline(terms)` / `latest_state(terms)` - every matching claim by `(date, seq)`, or the most recent one.
//! * `header(pairs)` - dated quotations only, oldest first; no instruction text.
//!
//! Everything is deterministic: same units + same query give the same pairs in the same order.

use std::collections::BTreeSet;

/// Verbs that make a first-person sentence an action/state claim ("I've completed", "we never set up").
const VERBS: &[&str] = &[
    "completed",
    "implemented",
    "created",
    "fixed",
    "written",
    "set up",
    "developed",
    "built",
    "configured",
    "tested",
    "solved",
    "attended",
    "met",
    "signed",
    "updated",
    "purchased",
    "taken",
];
const STOP: &[&str] = &[
    "the", "a", "an", "in", "on", "at", "of", "to", "for", "with", "by", "from", "is", "was",
    "are", "were", "and", "or", "but", "so", "that", "it", "this", "have", "has", "had", "been",
    "be",
];
/// Content words the question (or terms), the claim and the denial must all share.
pub const MIN_SHARED: usize = 2;

/// One user sentence that affirms (+1) or denies (-1) a first-person action or state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    pub unit_id: String,
    pub date: String,
    pub polarity: i8,
    pub text: String,
    pub words: BTreeSet<String>,
    /// Insertion order; `(date, seq)` is the ledger's time order. `unit_id` is never sorted on.
    pub seq: usize,
}

impl Claim {
    pub fn order(&self) -> (&str, usize) {
        (self.date.as_str(), self.seq)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePair {
    pub affirm: Claim,
    pub deny: Claim,
    pub shared: Vec<String>,
}

impl StatePair {
    /// Length-normalised overlap: |shared|^2 / sqrt(|affirm.words| * |deny.words|).
    pub fn strength(&self) -> f64 {
        pair_strength(
            self.shared.len(),
            self.affirm.words.len(),
            self.deny.words.len(),
        )
    }
}

pub fn pair_strength(shared: usize, affirm_words: usize, deny_words: usize) -> f64 {
    let denominator = ((affirm_words.max(1) * deny_words.max(1)) as f64).sqrt();
    (shared * shared) as f64 / denominator
}

/// Inclusive ISO-date window for the programmatic path. Undated claims pass any window.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Window {
    pub since: Option<String>,
    pub until: Option<String>,
    pub min_shared: Option<usize>,
    pub limit: Option<usize>,
}

/// Lower-cased letter runs of three or more characters, stop-words removed (Python: `\b[a-z]{3,}\b`).
pub fn content_words(text: &str) -> BTreeSet<String> {
    let lower = text.to_lowercase();
    let mut out = BTreeSet::new();
    for run in lower.split(|c: char| !c.is_ascii_lowercase()) {
        if run.len() >= 3 && !STOP.contains(&run) {
            out.insert(run.to_string());
        }
    }
    out
}

/// Words of a sentence for rule matching: lower-cased, split on whitespace, outer punctuation trimmed, inner
/// apostrophes kept ("i've", "haven't"). Mirrors the `\b`-bounded, case-insensitive Python patterns.
fn tokens(sentence: &str) -> Vec<String> {
    sentence
        .to_lowercase()
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Match the subject of a claim at `i`: "i", "i've", "i have", "we", "we've", "we have". Returns the next index.
fn subject(t: &[String], i: usize) -> Option<usize> {
    match t.get(i).map(String::as_str) {
        Some("i") | Some("we") => {
            if t.get(i + 1).map(String::as_str) == Some("have") {
                Some(i + 2)
            } else {
                Some(i + 1)
            }
        }
        Some("i've") | Some("we've") => Some(i + 1),
        _ => None,
    }
}

/// Match one of `VERBS` at `i` (two-word "set up" included). Returns the next index.
fn verb(t: &[String], i: usize) -> Option<usize> {
    let w = t.get(i)?.as_str();
    if w == "set" && t.get(i + 1).map(String::as_str) == Some("up") {
        return Some(i + 2);
    }
    if VERBS.contains(&w) {
        Some(i + 1)
    } else {
        None
    }
}

fn is_affirm(t: &[String]) -> bool {
    for i in 0..t.len() {
        if let Some(j) = subject(t, i) {
            if verb(t, j).is_some() {
                return true; // SUBJ VERB
            }
        }
        let w = t[i].as_str();
        if (w == "i" || w == "we")
            && matches!(
                t.get(i + 1).map(String::as_str),
                Some("always") | Some("usually")
            )
        {
            return true; // (i|we) (always|usually)
        }
        if w == "yes" && subject(t, i + 1).is_some() {
            return true; // yes, SUBJ
        }
    }
    false
}

fn is_deny(t: &[String]) -> bool {
    for i in 0..t.len() {
        if let Some(j) = subject(t, i) {
            // SUBJ (never|not yet) VERB
            let k = match t.get(j).map(String::as_str) {
                Some("never") => Some(j + 1),
                Some("not") if t.get(j + 1).map(String::as_str) == Some("yet") => Some(j + 2),
                _ => None,
            };
            if let Some(k) = k {
                if verb(t, k).is_some() {
                    return true;
                }
            }
        }
        let w = t[i].as_str();
        if (w == "i" || w == "we")
            && matches!(
                t.get(i + 1).map(String::as_str),
                Some("never") | Some("rarely")
            )
        {
            return true; // (i|we) (never|rarely)
        }
        // (haven't|have not|didn't|did not|never) VERB
        let k = match w {
            "haven't" | "didn't" | "never" => Some(i + 1),
            "have" | "did" if t.get(i + 1).map(String::as_str) == Some("not") => Some(i + 2),
            _ => None,
        };
        if let Some(k) = k {
            if verb(t, k).is_some() {
                return true;
            }
        }
        // refused to | decided against | avoided
        if w == "avoided"
            || (w == "refused" && t.get(i + 1).map(String::as_str) == Some("to"))
            || (w == "decided" && t.get(i + 1).map(String::as_str) == Some("against"))
        {
            return true;
        }
    }
    false
}

/// The question gate: `^(?:have (?:i|you)|has [a-z]+|do i|did i|how experienced am i)\b`, case-insensitive.
pub fn question_gate(question: &str) -> bool {
    let t = tokens(question.trim());
    let w = |i: usize| t.get(i).map(String::as_str);
    match w(0) {
        Some("have") => matches!(w(1), Some("i") | Some("you")),
        Some("has") => {
            w(1).is_some_and(|x| !x.is_empty() && x.chars().all(|c| c.is_ascii_lowercase()))
        }
        Some("do") | Some("did") => w(1) == Some("i"),
        Some("how") => w(1) == Some("experienced") && w(2) == Some("am") && w(3) == Some("i"),
        _ => false,
    }
}

/// Sentences of a unit spoken by the user. A role tag (`user:` / `assistant:` / `system:`) opens a speaker's span;
/// untagged lines continue the current speaker; a unit with no tags at all is user speech.
fn user_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut speaker = "user".to_string();
    let mut span: Vec<String> = Vec::new();
    let flush = |speaker: &str, span: &mut Vec<String>, out: &mut Vec<String>| {
        if speaker == "user" {
            out.extend(split_sentences(&span.join(" ")));
        }
        span.clear();
    };
    for line in text.lines() {
        if let Some((role, rest)) = role_tag(line) {
            flush(&speaker, &mut span, &mut out);
            speaker = role;
            span.push(rest.trim().to_string());
        } else if !line.trim().is_empty() {
            span.push(line.trim().to_string());
        }
    }
    flush(&speaker, &mut span, &mut out);
    out
}

fn role_tag(line: &str) -> Option<(String, &str)> {
    let trimmed = line.trim_start();
    for role in ["user", "assistant", "system"] {
        if trimmed.len() >= role.len() && trimmed[..role.len()].eq_ignore_ascii_case(role) {
            let rest = trimmed[role.len()..].trim_start();
            if let Some(after) = rest.strip_prefix(':') {
                return Some((role.to_string(), after.trim_start()));
            }
        }
    }
    None
}

/// `(?<=[.!?])\s+(?=\S)`: split after sentence punctuation followed by whitespace.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        current.push(c);
        if matches!(c, '.' | '!' | '?') && chars.get(i + 1).is_some_and(|n| n.is_whitespace()) {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() {
                out.push(std::mem::take(&mut current));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Index user claims from dialogue units; pair an affirmation with a denial when asked.
#[derive(Debug, Default, Clone)]
pub struct StateLedger {
    pub claims: Vec<Claim>,
}

impl StateLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index one unit. Returns the number of claims recorded from it.
    pub fn add(&mut self, unit_id: &str, text: &str, date: &str) -> usize {
        let before = self.claims.len();
        for sentence in user_sentences(text) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let t = tokens(sentence);
            let (aff, den) = (is_affirm(&t), is_deny(&t));
            if aff == den {
                continue; // neither, or both in one sentence: not a clean claim
            }
            self.claims.push(Claim {
                unit_id: unit_id.to_string(),
                date: date.to_string(),
                polarity: if aff { 1 } else { -1 },
                text: sentence.to_string(),
                words: content_words(sentence),
                seq: self.claims.len(),
            });
        }
        self.claims.len() - before
    }

    fn pairs(&self, q: &BTreeSet<String>, w: &Window) -> Vec<StatePair> {
        let min_shared = w.min_shared.unwrap_or(MIN_SHARED);
        if q.len() < min_shared {
            return Vec::new();
        }
        let in_window = |c: &Claim| in_window(&c.date, w.since.as_deref(), w.until.as_deref());
        let overlap =
            |c: &Claim| -> BTreeSet<String> { q.intersection(&c.words).cloned().collect() };
        let aff: Vec<(&Claim, BTreeSet<String>)> = self
            .claims
            .iter()
            .filter(|c| c.polarity == 1 && in_window(c))
            .map(|c| (c, overlap(c)))
            .filter(|(_, o)| o.len() >= min_shared)
            .collect();
        let den: Vec<(&Claim, BTreeSet<String>)> = self
            .claims
            .iter()
            .filter(|c| c.polarity == -1 && in_window(c))
            .map(|c| (c, overlap(c)))
            .filter(|(_, o)| o.len() >= min_shared)
            .collect();
        let mut pairs = Vec::new();
        for (a, ao) in &aff {
            for (d, dobj) in &den {
                let shared: Vec<String> = ao.intersection(dobj).cloned().collect();
                if shared.len() >= min_shared && a.unit_id != d.unit_id {
                    pairs.push(StatePair {
                        affirm: (*a).clone(),
                        deny: (*d).clone(),
                        shared,
                    });
                }
            }
        }
        pairs.sort_by(|x, y| {
            y.strength()
                .partial_cmp(&x.strength())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.affirm.seq.cmp(&y.affirm.seq))
                .then(x.deny.seq.cmp(&y.deny.seq))
        });
        pairs
    }

    /// Prompt path: empty unless the question passes the gate. Strongest pair first, then ledger order.
    pub fn conflicts_for(&self, question: &str) -> Vec<StatePair> {
        if !question_gate(question) {
            return Vec::new();
        }
        self.pairs(&content_words(question), &Window::default())
    }

    /// Programmatic path: no gate; inclusive date window, threshold and limit from `window`.
    pub fn conflicts_for_entity(&self, terms: &[&str], window: &Window) -> Vec<StatePair> {
        let mut pairs = self.pairs(&terms_words(terms), window);
        if let Some(limit) = window.limit {
            pairs.truncate(limit);
        }
        pairs
    }

    /// Every claim of either polarity sharing >= `min_shared` (default 1) words with `terms`, by `(date, seq)`.
    pub fn timeline(&self, terms: &[&str], window: &Window) -> Vec<Claim> {
        let q = terms_words(terms);
        let min_shared = window.min_shared.unwrap_or(1);
        let mut hits: Vec<Claim> = self
            .claims
            .iter()
            .filter(|c| q.intersection(&c.words).count() >= min_shared)
            .filter(|c| in_window(&c.date, window.since.as_deref(), window.until.as_deref()))
            .cloned()
            .collect();
        hits.sort_by(|a, b| a.order().cmp(&b.order()));
        hits
    }

    /// The most recent claim on the topic, either polarity, by `(date, seq)`.
    pub fn latest_state(&self, terms: &[&str], window: &Window) -> Option<Claim> {
        self.timeline(terms, window).pop()
    }
}

fn terms_words(terms: &[&str]) -> BTreeSet<String> {
    terms.iter().flat_map(|t| content_words(t)).collect()
}

fn in_window(date: &str, since: Option<&str>, until: Option<&str>) -> bool {
    if date.is_empty() {
        return true;
    }
    since.is_none_or(|s| date >= s) && until.is_none_or(|u| date <= u)
}

/// The text placed before retrieved context: both statements of each pair, dated, oldest first, nothing else.
pub fn header(pairs: &[StatePair], max_pairs: usize, max_words: usize) -> String {
    let mut lines = Vec::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for p in pairs.iter().take(max_pairs) {
        let mut both = [&p.affirm, &p.deny];
        both.sort_by(|a, b| a.order().cmp(&b.order()));
        for c in both {
            if !seen.insert((c.unit_id.clone(), c.text.clone())) {
                continue;
            }
            let words: Vec<&str> = c.text.split_whitespace().collect();
            let mut quote = words
                .iter()
                .take(max_words)
                .copied()
                .collect::<Vec<_>>()
                .join(" ");
            if words.len() > max_words {
                quote.push_str(" ...");
            }
            let when = if c.date.is_empty() {
                String::new()
            } else {
                format!("On {}, ", c.date)
            };
            lines.push(format!("{when}user said: \"{quote}\""));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> StateLedger {
        let mut led = StateLedger::new();
        led.add(
            "0:1",
            "user: I've written integration tests for the socket events, they run on every push.",
            "2023-05-11",
        );
        led.add("3:2", "user: I've never written integration tests for any socket events, we only have unit tests.", "2023-08-02");
        led.add(
            "4:0",
            "assistant: I've completed the deployment for you.\nuser: thanks",
            "2023-08-03",
        );
        led.add("5:0", "user: I usually deploy on Fridays.", "2023-09-01");
        led
    }

    fn windowed() -> StateLedger {
        let mut led = StateLedger::new();
        led.add(
            "10:2",
            "user: I've completed the migration runbook for the billing service.",
            "2024-03-26",
        );
        led.add("2:1", "user: I've never completed the migration runbook for the billing service, it is still a draft.", "2024-03-26");
        led.add("7:0", "user: Yes, I've completed the migration runbook for the billing service and the payments service and the ledger service and the archive service and the reporting service and the audit service too.", "2024-05-01");
        led.add(
            "9:0",
            "user: I usually deploy the billing service on Fridays.",
            "",
        );
        led
    }

    #[test]
    fn pairs_affirmation_with_later_denial() {
        let pairs = ledger().conflicts_for("Have I written integration tests for socket events?");
        assert_eq!(pairs.len(), 1);
        assert_eq!(
            (
                pairs[0].affirm.unit_id.as_str(),
                pairs[0].deny.unit_id.as_str()
            ),
            ("0:1", "3:2")
        );
        for w in ["integration", "tests", "socket", "events"] {
            assert!(pairs[0].shared.iter().any(|s| s == w), "{w}");
        }
    }

    #[test]
    fn assistant_lines_are_not_claims() {
        assert!(ledger().claims.iter().all(|c| c.unit_id != "4:0"));
    }

    #[test]
    fn question_gate_blocks_other_shapes() {
        let led = ledger();
        for q in [
            "What did I say about integration tests?",
            "Summarize our discussion of socket events and tests.",
            "When did I set up integration tests for socket events?",
            "Which tests cover the socket events?",
        ] {
            assert!(!question_gate(q), "{q}");
            assert!(led.conflicts_for(q).is_empty());
        }
        assert!(question_gate("Has anyone completed it?"));
        assert!(question_gate("How experienced am I with sockets?"));
    }

    #[test]
    fn header_quotes_both_dated_with_no_instruction_text() {
        let h = header(
            &ledger().conflicts_for("Have I written integration tests for socket events?"),
            2,
            40,
        );
        let lines: Vec<&str> = h.lines().collect();
        assert!(lines[0].starts_with("On 2023-05-11, user said: \""));
        assert!(lines[1].starts_with("On 2023-08-02, user said: \""));
        let lowered = h.to_lowercase();
        for banned in [
            "conflict",
            "note",
            "possible",
            "[",
            "]",
            "contradict",
            "please",
            "answer",
            "consider",
        ] {
            assert!(!lowered.contains(banned), "{banned}");
        }
        assert_eq!(header(&[], 2, 40), "");
    }

    #[test]
    fn seq_follows_insertion_and_orders_same_date_header() {
        let led = windowed();
        assert_eq!(
            led.claims.iter().map(|c| c.seq).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(led.claims[0].order() < led.claims[1].order());
        let h = header(
            &led.conflicts_for("Have I completed the migration runbook for the billing service?"),
            2,
            40,
        );
        assert!(h
            .lines()
            .next()
            .unwrap()
            .ends_with("\"I've completed the migration runbook for the billing service.\""));
    }

    #[test]
    fn ranking_prefers_tight_flip_over_long_sentence() {
        let pairs = windowed()
            .conflicts_for("Have I completed the migration runbook for the billing service?");
        let ids: Vec<&str> = pairs.iter().map(|p| p.affirm.unit_id.as_str()).collect();
        assert_eq!(ids, vec!["10:2", "7:0", "9:0"]);
        assert!(
            pairs[0].strength() > pairs[1].strength() && pairs[1].strength() > pairs[2].strength()
        );
        assert_eq!(pairs[0].shared.len(), pairs[1].shared.len());
    }

    #[test]
    fn entity_window_threshold_and_limit() {
        let led = windowed();
        let terms = ["migration runbook", "billing service"];
        let ids = |ps: Vec<StatePair>| {
            ps.iter()
                .map(|p| p.affirm.unit_id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(led.conflicts_for_entity(&terms, &Window::default())),
            vec!["10:2", "7:0", "9:0"]
        );
        assert_eq!(
            led.conflicts_for_entity(
                &terms,
                &Window {
                    limit: Some(1),
                    ..Default::default()
                }
            )
            .len(),
            1
        );
        let until = Window {
            until: Some("2024-03-31".into()),
            ..Default::default()
        };
        assert_eq!(
            ids(led.conflicts_for_entity(&terms, &until)),
            vec!["10:2", "9:0"]
        );
        let since = Window {
            since: Some("2024-04-01".into()),
            ..Default::default()
        };
        assert!(led.conflicts_for_entity(&terms, &since).is_empty());
        let three = Window {
            min_shared: Some(3),
            ..Default::default()
        };
        assert_eq!(
            ids(led.conflicts_for_entity(&terms, &three)),
            vec!["10:2", "7:0"]
        );
        let five = Window {
            min_shared: Some(5),
            ..Default::default()
        };
        assert!(led.conflicts_for_entity(&terms, &five).is_empty());
    }

    #[test]
    fn timeline_and_latest_state_by_date_then_seq() {
        let led = windowed();
        let tl = led.timeline(&["billing service"], &Window::default());
        assert_eq!(
            tl.iter().map(|c| c.unit_id.as_str()).collect::<Vec<_>>(),
            vec!["9:0", "10:2", "2:1", "7:0"]
        );
        assert_eq!(
            led.latest_state(&["billing service"], &Window::default())
                .unwrap()
                .unit_id,
            "7:0"
        );
        let until = Window {
            until: Some("2024-03-31".into()),
            ..Default::default()
        };
        assert_eq!(
            led.latest_state(&["billing service"], &until)
                .unwrap()
                .unit_id,
            "2:1"
        );
        assert!(led
            .latest_state(&["nothing here"], &Window::default())
            .is_none());
    }

    #[test]
    fn content_words_match_python_shape() {
        let w = content_words("I don't have the Socket-events, it's a draft!");
        assert!(
            w.contains("don")
                && w.contains("socket")
                && w.contains("events")
                && w.contains("draft")
        );
        assert!(!w.contains("the") && !w.contains("it") && !w.contains("have"));
    }
}
