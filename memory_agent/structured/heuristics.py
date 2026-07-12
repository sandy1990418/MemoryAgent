"""Deterministic linguistic heuristics shared by the updater and verifier.

These regexes encode what counts as an exact value, a subject-bound value, a
status-change cue, or durable user state. They are deliberately separate from
the LLM-driven updater so the policy-sensitive vocabulary can be reviewed and
tuned in one place.
"""

from __future__ import annotations

import re

from memory_agent.models.policy import MemoryPolicy, is_chat_policy

MONTH_NAMES = (
    "Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    "Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
# Bare dates ("March 15, 2024") are useless without their subject: at answer
# time nobody can tell a deployment deadline from a sprint start. Date matches
# therefore get a same-sentence context prefix; self-describing values
# (versions, measured values, paths) do not need one.
EXACT_VALUE_DATE_PATTERNS = [
    re.compile(
        rf"\b(?:{MONTH_NAMES})\.?\s+\d{{1,2}},\s+\d{{4}}\s*-\s*"
        rf"(?:{MONTH_NAMES})\.?\s+\d{{1,2}},\s+\d{{4}}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:{MONTH_NAMES})\.?\s+\d{{1,2}}\s*-\s*"
        rf"(?:(?:{MONTH_NAMES})\.?\s+)?\d{{1,2}},\s+\d{{4}}\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?:{MONTH_NAMES})\.?\s+\d{{1,2}},\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]
EXACT_VALUE_PATTERNS = [
    re.compile(
        r"\b(?:Python|Flask(?:-Login|-SQLAlchemy|-Migrate|-WTF|-Argon2|-Talisman)?|"
        r"SQLite|Jinja2|Bootstrap|Chart\.js|Marshmallow|SQLAlchemy|Gunicorn|Redis|"
        r"PostgreSQL|Loggly|WCAG|flake8|black|pytest|bcrypt|Argon2)"
        r"\s+v?\d+(?:\.\d+){0,3}(?:\s+[A-Z]{1,3})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|MB|GB|fps|%)\b", re.IGNORECASE),
    re.compile(
        r"\b\d+(?:\.\d+)?\s+"
        r"(?:workers?|branches?|users?|failed login attempts?|attempts?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bport\s+\d{2,5}\b", re.IGNORECASE),
    re.compile(r"\b(?:pull request|PR)\s+#?\d+\b", re.IGNORECASE),
    re.compile(r"\bv\d+(?:\.\d+){1,3}\b", re.IGNORECASE),
    re.compile(r"(?<!\w)/(?:[\w.-]+/)*[\w.-]+"),
    re.compile(r"\b[A-Za-z_][\w.-]*\.(?:py|html|css|js|json|log|yml|yaml|md|txt)\b"),
    re.compile(
        r"\b(?:TemplateNotFound|OperationalError|KeyError|TypeError|ValueError)"
        r"(?::\s*['\"]?[\w.-]+['\"]?)?",
    ),
]
PERSONAL_SUBJECT_VALUE_PATTERNS = [
    re.compile(r"(?:[$€£]\s?\d+(?:,\d{3})*(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE),
    # A counted noun is not tied to any fixed vocabulary: "12 books",
    # "52 sources", "4,700 words", and "3 sessions" are the same shape. The
    # SUBJECT_VALUE_SECTION_RE cue gate keeps this from becoming an inventory.
    re.compile(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s+"
        r"(?:days?|weeks?|months?|years?|hours?|minutes?|[A-Za-z][a-z]{2,}s)\b",
        re.IGNORECASE,
    ),
]
# Personal values belong to the user's life (money, reading goals, study
# counts). Software-project metrics reaching memory deterministically is the
# practical/eval profiles' job via SUBJECT_VALUE_PATTERNS; for personal-value
# extraction a technical context means "not a personal value".
TECHNICAL_CONTEXT_RE = re.compile(
    r"\b(?:api|server|service|worker|endpoint|deploy(?:ment|ed)?|build|pipeline|"
    r"database|schema|coverage|latency|response time|integration|unit test|"
    r"tests?|codebase|repositor(?:y|ies)|branch(?:es)?)\b",
    re.IGNORECASE,
)
SUBJECT_VALUE_PATTERNS = [
    re.compile(
        r"\b\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*"
        r"(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*"
        r"(?:calls(?:/day| per day)?|project cards?|cards?|columns?|"
        r"features?|items?|days?|weeks?|attempts?|failed login attempts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:calls/day|calls per day|project cards?|cards?|columns?|"
        r"features?|items?|days?|weeks?|attempts?|failed login attempts?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|MB|GB|seconds?|minutes?)\b", re.IGNORECASE),
]
SUBJECT_VALUE_SECTION_RE = re.compile(
    r"\b(?:updated|changed|moved|new|now|latest|increased|decreased|reduced|"
    r"improved|completed|achieved|added|reached|coverage|quota|deadline|"
    r"count|total|cards?|columns?|calls?|latency|response time|"
    r"rate(?: limit)?|budget|aim(?:ing)?|goal|target|duration|takes?|test)\b",
    re.IGNORECASE,
)
STATUS_VALUE_RE = re.compile(
    r"\b(?:updated|changed|moved|new|now|latest|increased|decreased|reduced)\b",
    re.IGNORECASE,
)
PROGRESS_VALUE_RE = re.compile(
    r"\b(?:completed|implemented|fixed|achieved|improved|reduced|coverage|"
    r"latency|response time|test)\b",
    re.IGNORECASE,
)
# Shared by the deterministic status-change extractor (updater) and by
# MemoryUpdateVerifier (memory_agent/structured/verifier.py). Keep them on the
# same regex: if the verifier recognized cues the extractor does not, every
# such turn would fail verification forever (the extractor never records it,
# so retries can never satisfy the check). CJK cues carry no \b word
# boundaries because they do not tokenize on word characters.
STATUS_CHANGE_CUE_RE = re.compile(
    r"\b(?:never|not anymore|no longer|changed my mind|actually|instead|"
    r"contradiction|contradictory|starting from scratch)\b"
    r"|(?:其實|不是|改成|不再|沒有|不要記|改用)",
    re.IGNORECASE,
)
# Practical profile: only unambiguous corrections/reversals. Broad cues like
# "actually"/"instead"/"其實"/"不是" fire on ordinary sentences and turn
# status_changes into a change-log; that is eval-profile behavior.
PRACTICAL_STATUS_CHANGE_CUE_RE = re.compile(
    r"\b(?:not anymore|no longer|changed my mind|correction|"
    r"contradiction|contradictory)\b"
    r"|(?:改成|不再|不要記|改用|更正)",
    re.IGNORECASE,
)
# A first-person denial of past experience is durable regardless of domain:
# it contradicts (or preempts) any affirmative record about the same subject,
# so both project verbs and common personal-history verbs are cues.
EXPLICIT_PROJECT_DENIAL_RE = re.compile(
    r"\bI(?:'ve| have) never (?:actually )?"
    r"(?:written|implemented|integrated|deployed|used|handled|managed|configured|"
    r"installed|enabled|created|built|completed|"
    r"met|read|visited|attended|listened|watched|seen|joined|finished|signed|"
    r"missed)\b",
    re.IGNORECASE,
)


def status_change_cue_re(policy: MemoryPolicy | None) -> re.Pattern[str]:
    """Policy-aware cue regex. The extractor and MemoryUpdateVerifier MUST both
    resolve cues through this helper: if the verifier recognized cues the
    extractor does not, those turns would fail verification on every retry."""
    if policy is not None and is_chat_policy(policy):
        return PRACTICAL_STATUS_CHANGE_CUE_RE
    return STATUS_CHANGE_CUE_RE


WHITESPACE_RE = re.compile(r"\s+")
CONTEXT_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
CONTEXT_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has",
    "user", "assistant", "about", "into", "should", "would", "could",
    "your", "you", "are", "was", "were", "been", "being", "not",
}
GENERIC_NON_DURABLE_MEMORY_RE = re.compile(
    r"\b(?:user\s+)?(?:asked|asks|inquired|wanted to know|discussed|talked)\s+"
    r"(?:about|whether|how|what|why|when)\b"
    r"|\btopic (?:was|is) discussed\b",
    re.IGNORECASE,
)
# Practical profile stores user-stated durable state only. Entries that
# attribute content to the assistant (advice, tutorials, proposed schedules)
# are dropped at the filter, not just discouraged in the prompt.
ASSISTANT_ATTRIBUTED_RE = re.compile(
    r"^\s*(?:the\s+)?assistant\b"
    r"|\bassistant(?:'s)?\s+(?:stated|said|suggested|recommended|proposed|"
    r"provided|created|advised|outlined|offered|explained|plan(?:s|ned)?|"
    r"schedule[sd]?|tutorial|example)\b",
    re.IGNORECASE,
)
DURABLE_USER_STATE_RE = re.compile(
    r"\b(?:"
    r"i(?:'m| am) (?:working on|having trouble with|trying to (?:implement|integrate))|"
    r"i (?:prefer|need|want|chose|decided|implemented|fixed|observed|got|hit|saw|am using|am working on|am having trouble with|am trying to (?:implement|integrate)|"
    r"switched|changed my mind|will use|do not want|don't want|cannot|can't)|"
    r"we (?:chose|decided|implemented|fixed|are using|will use|switched)|"
    r"always |from now on|going forward|for (?:this|the|our|my) project|"
    r"please (?:keep|use|avoid)|(?:do not|don't|never) (?:use|include|add)|"
    r"(?:answers?|responses?) should |"
    r"(?:the |our |my )?(?:project|app|application|build|deployment|tests?|"
    r"implementation|integration|pipeline|service|api|repository|branch) "
    r"(?:is|are|uses|has|failed|fails|passed|passes|returns|blocks?|needs?)|"
    r"(?:error|exception|failure|blocker) (?:is|was|occurs?|says?)|"
    r"(?:failed|tried|attempted) (?:to|using)|blocked (?:by|on)|"
    r"use .{1,80} (?:instead of|rather than)|"
    r"(?:correction|not anymore|no longer|changed my mind)"
    r")\b",
    re.IGNORECASE,
)
STABLE_INSTRUCTION_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:always\s+(?:format|include|provide|use|avoid|keep)|"
    r"from now on|going forward|please (?:keep|use|avoid)|"
    r"(?:answers?|responses?|code snippets?) should)\b",
    re.IGNORECASE,
)
PROJECT_IMPLEMENTATION_STATE_RE = re.compile(
    r"\b(?:i(?:'ve| have) (?:already )?(?:implemented|integrated|completed|managed to)|"
    r"we(?:'ve| have) (?:implemented|integrated|completed)|"
    r"i(?:'m| am) (?:working on|having trouble with|trying to "
    r"(?:implement|integrate|configure|set up|optimize|debug|fix|document|plan))|"
    r"i (?:implemented|integrated|completed|configured|deployed|fixed|resolved|"
    r"documented|added|tested)|"
    r"we (?:implemented|integrated|completed|configured|deployed|fixed|resolved|"
    r"documented|added|tested))\b",
    re.IGNORECASE,
)
ORDINARY_QUESTION_RE = re.compile(
    r"^\s*(?:how|what|why|when|where|who|which|can|could|would|should|is|are|"
    r"do|does|did|explain|translate|show|give|tell)\b",
    re.IGNORECASE,
)


# Chat turns often wrap one durable statement in a conversational frame:
# "I'm kinda worried that <state>, can you help me <request>?". The frame adds
# tokens without adding memory value. Trimming never removes a span that
# carries a number, currency, or percent — those may be the value itself.
_LEADING_FRAME_RE = re.compile(
    r"^(?:(?:so|but|and|well|ok(?:ay)?)[,\s]+)?"
    r"i'?m\s+(?:kinda\s+|really\s+|a\s+bit\s+|just\s+)?"
    r"(?:worried|concerned|wondering|curious|thinking|torn|confused|unsure|not\s+sure)\s*"
    r"(?:that\s+|about\s+(?:how\s+|whether\s+|if\s+)?|if\s+|whether\s+|how\s+)?",
    re.IGNORECASE,
)
_TRAILING_REQUEST_RE = re.compile(
    r"[,;]?\s*(?:(?:so|but|and)\s+)?"
    r"(?:(?:can|could|would|will|should)\s+you\b"
    r"|do\s+you\s+think\b|what(?:'s|\s+is)\s+the\s+best\s+way\b"
    r"|any\s+(?:suggestions?|advice|thoughts?)\b"
    r"|(?:so\s+)?i'?m\s+wondering\s+if\b"
    r"|how\s+(?:should|can|do)\s+i\b).*$",
    re.IGNORECASE,
)
_VALUE_BEARING_RE = re.compile(r"[\d$€£%]")


def trim_conversational_frame(text: str) -> str:
    """Strip question framing around a durable statement, keeping any values."""
    trimmed = text
    trailing = _TRAILING_REQUEST_RE.search(trimmed)
    if trailing and not _VALUE_BEARING_RE.search(trailing.group(0)):
        candidate = trimmed[: trailing.start()].rstrip(" ,;")
        if len(candidate) >= 20:
            trimmed = candidate
    leading = _LEADING_FRAME_RE.match(trimmed)
    if leading and not _VALUE_BEARING_RE.search(leading.group(0)):
        candidate = trimmed[leading.end():].lstrip()
        if len(candidate) >= 20:
            trimmed = candidate
    return trimmed


def content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in CONTEXT_WORD_RE.findall(text)
        if len(word) >= 3 and word.lower() not in CONTEXT_STOPWORDS
    }
