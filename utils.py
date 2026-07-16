"""
Pure helper functions for the K-12 education contact scraper.

No scraping logic here — just typed, reusable utilities for
validating, normalizing, and deduplicating contact data.
"""

from __future__ import annotations

import re

# Simple RFC-5322-ish pattern: good enough for validating scraped emails
# without pulling in a heavier dependency.
_EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)
_EMAIL_EXTRACT_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Generic link-label prefixes mistakenly merged into scraped person names
_NAME_LINK_LABEL_PREFIXES: tuple[str, ...] = (
    "email",
    "e-mail",
    "contact",
    "view profile",
    "profile",
    "send email",
    "mail",
    "message",
)

# Whole-string group/department labels that are never person names
_GROUP_LABEL_EXACT: frozenset[str] = frozenset({
    "business office",
    "committee members",
    "main office",
    "central office",
    "district office",
    "school office",
    "food service",
    "transportation",
    "facilities",
    "human resources",
    "it",
    "technology",
    "front office",
    "finance",
})

_GENERIC_PAGE_LABELS: frozenset[str] = frozenset({
    "contact us",
    "about us",
    "staff directory",
    "district office",
    "home",
    "welcome",
    "directory",
    "staff",
})

# Reject names ending with these department/group terms (whole-string suffix)
_GROUP_LABEL_ENDINGS: tuple[str, ...] = (
    "office",
    "committee",
    "members",
    "department",
    "staff",
)

# Nav/label junk stripped from scraped job titles
_TITLE_JUNK_PREFIXES: tuple[str, ...] = (
    "titles:",
    "title:",
    "locations:",
    "location:",
    "roles:",
    "role:",
)

_TITLE_JUNK_SUFFIXES: tuple[str, ...] = (
    "locations",
    "location",
    "titles",
    "title",
)

_TITLE_JUNK_EXACT: frozenset[str] = frozenset({
    "locations",
    "location",
    "titles",
    "title",
})

_TRAILING_PHONE_IN_PARENS = re.compile(
    r"\s*\(\s*(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}"
    r"(?:\s*(?:ext\.?|x)\s*\d+)?\s*\)\s*$",
    re.IGNORECASE,
)
_TRAILING_PHONE_BARE = re.compile(
    r"\s+(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}"
    r"(?:\s*(?:ext\.?|x)\s*\d+)?\s*$",
    re.IGNORECASE,
)
# Inline phone + extension chunks (no "Phone:" prefix required)
_INLINE_PHONE_CHUNK = re.compile(
    r"\s+\d{3}-\d{3}-\d{4}(?:\s*[-–—]\s*Ext\.?\s*\d+)?",
    re.IGNORECASE,
)
_INLINE_EXT_SUFFIX = re.compile(r"\s*[-–—]\s*Ext\.?\s*\d+\s*", re.IGNORECASE)

# Review-column name shape: 2–4 Title Case words
_REVIEW_NAME_SHAPE = re.compile(
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$"
)


def _strip_trailing_punctuation(text: str) -> str:
    """Strip trailing punctuation without using rstrip character-class ranges."""
    return re.sub(r"[:.,;-]+$", "", text.strip())


def is_generic_page_label(text: str) -> bool:
    """Return True for nav/section headings mistaken for titles or organizations."""
    return clean_text(text).lower() in _GENERIC_PAGE_LABELS


def sanitize_scraped_email(raw: str) -> str:
    """
    Extract a clean email from scraped text, stripping JS placeholder junk.

    Handles values like ``jjoyce@domain.orgundefinedundefined`` where missing
    JS fields were concatenated as the literal word "undefined".
    """
    if not raw:
        return ""

    text = str(raw).strip()
    text = re.sub(r"(?i)(undefined|null|none)+", "", text)
    text = text.split("?")[0].strip()

    match = _EMAIL_EXTRACT_PATTERN.search(text)
    if match:
        candidate = match.group(0).lower()
        if _EMAIL_PATTERN.match(candidate):
            return candidate

    lowered = text.lower()
    return lowered if _EMAIL_PATTERN.match(lowered) else ""


def clean_person_name(raw: str) -> str:
    """Strip generic link-label prefixes from a scraped person name."""
    name = clean_text(str(raw))
    if not name:
        return ""

    changed = True
    while changed:
        changed = False
        lower = name.lower()
        for label in _NAME_LINK_LABEL_PREFIXES:
            for prefix in (f"{label} ", f"{label}: "):
                if lower.startswith(prefix):
                    name = name[len(prefix):].strip()
                    changed = True
                    break

    return name


def is_group_or_department_label(name: str) -> bool:
    """Return True when a candidate name is a department/group label, not a person."""
    lower = clean_text(name).lower()
    if not lower:
        return False
    if lower in _GROUP_LABEL_EXACT:
        return True
    return any(
        lower.endswith(f" {ending}") or lower == ending
        for ending in _GROUP_LABEL_ENDINGS
    )


def name_matches_review_shape(name: str) -> bool:
    """Return True when a name looks like a clean 2–4 word person name."""
    cleaned = clean_person_name(name)
    if not cleaned:
        return False
    return bool(_REVIEW_NAME_SHAPE.match(cleaned))


def name_email_pairing_confidence(name: str, email: str) -> str:
    """
    Score how confidently a scraped name belongs to an email address.

    Returns ``certain``, ``borderline``, or ``invalid``. Callers should drop
    rows marked ``invalid`` and flag ``borderline`` for manual review.
    """
    cleaned_name = clean_person_name(name)
    cleaned_email = sanitize_scraped_email(email)
    if not cleaned_name or not cleaned_email or "@" not in cleaned_email:
        return "invalid"

    local = cleaned_email.split("@", 1)[0].lower()
    local_compact = local.replace(".", "").replace("_", "")
    parts = [part.lower() for part in cleaned_name.split() if len(part) >= 2]
    if not parts:
        return "invalid"

    first, last = parts[0], parts[-1]
    last_present = (
        last in local_compact
        or f".{last}" in local
        or local.endswith(last)
        or local.startswith(last)
    )
    first_present = (
        first in local_compact
        or local_compact.startswith(first[:3])
        or local.startswith(first)
    )

    for part in parts:
        if part in local_compact or local_compact in part:
            return "certain"
        if len(part) >= 4 and part[:4] in local_compact:
            return "certain"

    if len(parts) >= 2:
        combos = (
            f"{first[0]}{last}",
            f"{first}{last[0]}",
            f"{first}{last}",
            f"{last}{first[0]}",
            f"{last}{first}",
        )
        for combo in combos:
            if combo and combo in local_compact:
                return "certain"

    if not last_present and not first_present:
        return "invalid"

    if last_present or first_present:
        return "borderline"

    return "invalid"


def title_needs_review(title: str) -> bool:
    """Return True when a job title has punctuation patterns worth spot-checking."""
    cleaned = clean_text(title)
    if not cleaned:
        return False
    if "|" in cleaned:
        return True
    separator_hits = len(re.findall(r"[,;]", cleaned))
    separator_hits += len(re.findall(r"\s[-–—]\s", cleaned))
    return separator_hits >= 2


def compute_needs_review(contact: dict) -> bool:
    """Return True when a contact row should be manually spot-checked."""
    if contact.get("pairing_confidence") == "borderline":
        return True
    if not name_matches_review_shape(contact.get("full_name", "")):
        return True
    if title_needs_review(contact.get("job_title", "")):
        return True
    return False


def clean_job_title_text(title: str) -> str:
    """
    Normalize a scraped job title, stripping phones, labels, and nav junk.

    Removes trailing phone numbers (including parenthetical), anything from
    ``Phone:`` onward, and generic prefixes/suffixes like ``Titles:`` or
    ``Locations``.
    """
    cleaned = clean_text(title)
    if not cleaned:
        return ""

    parts = re.split(r"\bPhone:\s*", cleaned, maxsplit=1, flags=re.IGNORECASE)
    cleaned = parts[0].strip()
    cleaned = _strip_trailing_punctuation(cleaned)

    changed = True
    while changed:
        changed = False
        updated = _TRAILING_PHONE_IN_PARENS.sub("", cleaned).strip()
        updated = _TRAILING_PHONE_BARE.sub("", updated).strip()
        updated = _INLINE_PHONE_CHUNK.sub("", updated).strip()
        updated = _INLINE_EXT_SUFFIX.sub("", updated).strip()
        updated = _strip_trailing_punctuation(updated)
        if updated != cleaned:
            cleaned = updated
            changed = True

    lower = cleaned.lower()
    if lower in _TITLE_JUNK_EXACT or is_generic_page_label(cleaned):
        return ""

    changed = True
    while changed:
        changed = False
        lower = cleaned.lower()
        for prefix in _TITLE_JUNK_PREFIXES:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                changed = True
                break
        lower = cleaned.lower()
        for suffix in _TITLE_JUNK_SUFFIXES:
            if re.search(rf"\b{re.escape(suffix)}\s*$", lower):
                cleaned = re.sub(
                    rf"\b{re.escape(suffix)}\s*$",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                ).strip()
                changed = True
                break

    return _strip_trailing_punctuation(cleaned)


def is_valid_email(email: str) -> bool:
    """
    Check whether a string looks like a valid email address.

    Args:
        email: The email string to validate.

    Returns:
        True if the string matches a standard email pattern, False otherwise.
    """
    if not email:
        return False
    cleaned = sanitize_scraped_email(email) or email.strip()
    return bool(_EMAIL_PATTERN.match(cleaned))


def normalize_phone(phone: str) -> str:
    """
    Normalize a phone number to a readable "(XXX) XXX-XXXX" format.

    Strips all non-digit characters first, then reformats. Falls back to
    returning the stripped digits if the number isn't a standard 10-digit
    (optionally 11-digit with US country code) number.

    Args:
        phone: The raw phone number string (may contain punctuation, spaces).

    Returns:
        A formatted phone number, or the raw digit string if it doesn't
        match a recognizable length, or an empty string if input is empty.
    """
    if not phone:
        return ""

    digits = re.sub(r"\D", "", phone)

    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

    return digits


def dedupe_key(name: str, org: str) -> str:
    """
    Build a normalized, lowercase key for deduplicating contacts.

    Args:
        name: The contact's full name.
        org: The contact's organization / school district.

    Returns:
        A "name|org" key with whitespace collapsed and text lowercased,
        suitable for use as a dict/set key when deduplicating.
    """
    name_part = clean_text(name).lower()
    org_part = clean_text(org).lower()
    return f"{name_part}|{org_part}"


def clean_text(text: str) -> str:
    """
    Clean a text string by stripping surrounding whitespace and
    collapsing internal whitespace/newlines into single spaces.

    Args:
        text: The raw text to clean.

    Returns:
        A whitespace-normalized string, or an empty string if input is empty.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# Organization / program terms that must not appear in person names or job titles
ORG_PROGRAM_SUBSTRINGS: tuple[str, ...] = (
    "school",
    "elementary",
    "home",
    "corrections",
    "institute",
    "directory",
    "project",
    "department",
    "district",
    "program",
    "special education",
    "academy",
    "center",
    "public schools",
    "schools",
    "isd",
)

# Verb-like tokens that indicate a bio sentence fragment, not a name or title
_SENTENCE_FRAGMENT_WORDS: frozenset[str] = frozenset({
    "has", "worked", "working", "spent", "career", "serves",
})
_SENTENCE_FRAGMENT_PHRASES: tuple[str, ...] = ("is a",)

_MAX_PERSON_NAME_WORDS = 6
_MAX_JOB_TITLE_WORDS = 12

# 2–3 Title Case words; hyphens and apostrophes allowed within a word
_NAME_WORD = r"[A-Z][a-z]*(?:[-'][A-Z][a-z]+)*"
_PERSON_NAME_SHAPE = re.compile(
    rf"^{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,2}}$",
)


def contains_org_program_language(text: str) -> bool:
    """Return True if text contains organization or program-like substrings."""
    lower = clean_text(text).lower()
    if not lower:
        return False
    return any(term in lower for term in ORG_PROGRAM_SUBSTRINGS)


def contains_sentence_fragment(text: str, *, include_career: bool = True) -> bool:
    """Return True if text reads like a sentence fragment rather than a label."""
    lower = clean_text(text).lower()
    if not lower:
        return False

    words = set(re.findall(r"[a-z']+", lower))
    fragment_words = _SENTENCE_FRAGMENT_WORDS
    if not include_career:
        fragment_words = fragment_words - {"career"}
    if words & fragment_words:
        return True

    return any(phrase in lower for phrase in _SENTENCE_FRAGMENT_PHRASES)


def is_valid_person_name(name: str) -> bool:
    """
    Check whether a candidate full_name looks like a person, not junk text.

    Rejects organization names, facility labels, program titles, and bio
    sentence fragments. Contacts that fail should be dropped entirely.

    Checks (in order):
      1. Empty or longer than six words
      2. Organization / program substrings
      3. Sentence-fragment verb tokens
      4. Basic person-name shape (2–3 capitalized words; only hyphens/apostrophes)

    Args:
        name: The candidate full name extracted from a page.

    Returns:
        True only if the string looks like a real person's name.
    """
    cleaned = clean_text(name)
    if not cleaned:
        return False

    words = cleaned.split()
    if len(words) > _MAX_PERSON_NAME_WORDS:
        return False

    if contains_org_program_language(cleaned):
        return False

    if contains_sentence_fragment(cleaned, include_career=True):
        return False

    if is_group_or_department_label(cleaned):
        return False

    if any(ch.isdigit() for ch in cleaned):
        return False

    if re.search(r"[^A-Za-z\s'\-]", cleaned):
        return False

    return bool(_PERSON_NAME_SHAPE.match(cleaned))


def is_valid_job_title_text(title: str) -> bool:
    """
    Check whether a candidate job title looks like a real title, not junk text.

    Rejects navigation labels, program/grant names, and bio-style sentences.
    Contacts with invalid titles should be dropped entirely.

    Args:
        title: The candidate job title extracted from a page.

    Returns:
        True only if the string looks like a plausible job title.
    """
    cleaned = clean_text(title)
    if not cleaned:
        return False

    if len(cleaned.split()) > _MAX_JOB_TITLE_WORDS:
        return False

    if contains_org_program_language(cleaned):
        return False

    if contains_sentence_fragment(cleaned, include_career=False):
        return False

    if "|" in cleaned:
        return False

    if is_generic_page_label(cleaned):
        return False

    return True
