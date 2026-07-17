"""
Web scraper module for extracting contact info from school district websites.

Uses Playwright (async API) to load JavaScript-rendered pages. The browser
and context are kept alive across many page visits via an async context
manager, avoiding the overhead of relaunching a browser per page.

Contact extraction uses BeautifulSoup to parse staff directory patterns:
name/title pairs, mailto: and tel: links, and LinkedIn profile anchors.
"""

from __future__ import annotations

import re
from types import TracebackType
from typing import NamedTuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from playwright.async_api import (
    Browser,
    BrowserContext,
    Playwright,
    async_playwright,
)

from districts import (
    STATE_ABBREVIATIONS,
    STATE_AREA_CODES,
    get_area_codes_for_state,
    get_conflict_hints_for_state,
    state_abbreviation,
)
from search import find_linkedin_url
from utils import (
    clean_job_title_text,
    clean_person_name,
    clean_text,
    compute_needs_review,
    contains_org_program_language,
    contains_sentence_fragment,
    is_generic_page_label,
    is_group_or_department_label,
    is_valid_email,
    is_valid_job_title_text,
    is_valid_person_name,
    name_email_pairing_confidence,
    normalize_phone,
    sanitize_scraped_email,
)

# Default navigation timeout (ms) for loading a page
_DEFAULT_TIMEOUT_MS = 15_000

# Loose job-title keywords for substring matching against directory text
DEFAULT_JOB_TITLE_KEYWORDS: list[str] = [
    "superintendent",
    "career",
    "cte",
    "college and career",
    "principal",
    "work-based learning",
    "workforce",
    "pathways",
    "postsecondary",
    "experiential learning",
]

# Coordinator/specialist titles require one of these qualifiers in the same title text
_CAREER_QUALIFIERS: tuple[str, ...] = (
    "career",
    "workforce",
    "pathways",
    "work-based learning",
    "college and career",
    "cte",
    "postsecondary",
    "experiential learning",
)

# Standalone occurrences of these in a title are not enough — need a career qualifier too
_CONDITIONAL_TITLE_TERMS: frozenset[str] = frozenset({"coordinator", "specialist"})

# Tags and class-name fragments that commonly wrap a single staff entry
_STAFF_CONTAINER_TAGS: set[str] = {"tr", "li", "article", "section", "div"}
_STAFF_CLASS_KEYWORDS: tuple[str, ...] = (
    "staff", "directory", "person", "employee", "member",
    "card", "profile", "contact", "team", "faculty",
)

# Scope-text fallback in _get_title_before_anchor: longer text is page junk, not a title
_CONTAINER_FALLBACK_TITLE_MAX_WORDS = 18
_MAX_JOB_TITLE_WORDS = 12

# Navigation / link text mistaken for job titles
_NAV_TITLE_PREFIXES: tuple[str, ...] = (
    "go to",
    "click here",
    "view",
    "return to",
    "back to",
)

# Fallback regex when mailto:/tel: links are absent
_EMAIL_IN_TEXT = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)
_PHONE_IN_TEXT = re.compile(
    r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}",
)

# Domains unlikely to yield reachable K-12 district contacts — skip parsing entirely
_EXCLUDED_DOMAIN_SUFFIXES: tuple[str, ...] = (
    "ed.gov",
    "whitehouse.gov",
    "gatesfoundation.org",
    "fordfoundation.org",
    "kelloggfoundation.org",
    "wkkf.org",
    "hewlett.org",
    "macfound.org",
    "carnegie.org",
    "wallacefoundation.org",
    "luminafoundation.org",
    "chanzuckerberg.com",
    "schottfoundation.org",
    "nea.org",
    "aasa.org",
    "ascd.org",
    "nsba.org",
    "ccsso.org",
    "educationweek.org",
    "wikipedia.org",
)


def _is_excluded_domain(url: str) -> bool:
    """
    Return True if a URL belongs to a domain that should be skipped.

    Matches federal sites, national policy pages, and well-known national
    foundations / associations that rarely list individual district contacts.
    """
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in _EXCLUDED_DOMAIN_SUFFIXES)


# Facilities / operations departments that produce false-positive leadership matches
_OPERATIONS_ROLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bof\s+building(?:s)?\b",
        r"\bof\s+grounds\b",
        r"\bof\s+transportation\b",
        r"\bof\s+facilities\b",
        r"\bof\s+maintenance\b",
        r"\bbuilding(?:s)?\s+and\s+grounds\b",
        r"\bbuildings?\s+&\s+grounds\b",
    )
)


def _clean_job_title(title: str) -> str:
    """Normalize a scraped job title."""
    return clean_job_title_text(title)


def _log_chosen_job_title(title: str, source: str) -> None:
    """Print the final job title chosen for one contact (debug)."""
    if title:
        print(f"[job_title chosen] {source}: {title!r}")


def _log_rejected_contact(
    reason: str,
    *,
    full_name: str = "",
    job_title: str = "",
    email: str = "",
    source: str = "",
) -> None:
    """Print one rejected contact with raw fields and the failing check (debug)."""
    print(
        f"[rejected: {reason}] source={source!r} "
        f"name={full_name!r} title={job_title!r} email={email!r}"
    )


def _full_name_rejection_reason(name: str) -> str | None:
    """Return a rejection reason when a scraped name is not a valid person."""
    cleaned = clean_person_name(name)
    if not cleaned:
        return "missing_name"
    if is_valid_person_name(cleaned):
        return None
    if contains_org_program_language(cleaned):
        return "org_program_name"
    if is_group_or_department_label(cleaned):
        return "group_label_name"
    return "invalid_name"


def _job_title_rejection_reason(title: str) -> str | None:
    """Return a rejection reason when a scraped title is not a valid job title."""
    title = clean_text(title)
    if not title:
        return "no_title_found"
    if "@" in title:
        return "invalid_title"
    if len(title.split()) > _MAX_JOB_TITLE_WORDS:
        return "title_too_long"
    if not is_valid_job_title_text(title):
        if is_generic_page_label(title):
            return "generic_label_title"
        if contains_org_program_language(title):
            return "org_program_title"
        if contains_sentence_fragment(title, include_career=False):
            return "sentence_fragment_title"
        return "invalid_title"
    lower = title.lower()
    for prefix in _NAV_TITLE_PREFIXES:
        if lower.startswith(prefix):
            return "nav_label_title"
    if lower == "home" or lower.endswith(" home"):
        words = title.split()
        if len(words) <= 4 and " of " not in lower:
            return "nav_label_title"
    return None


def _contact_rejection_reason(
    contact: dict,
    requested_job_titles: list[str] | None = None,
) -> str | None:
    """Return the first failing validation check for a contact, or None if keepable."""
    name_reason = _full_name_rejection_reason(contact.get("full_name", ""))
    if name_reason:
        return name_reason
    title_reason = _job_title_rejection_reason(contact.get("job_title", ""))
    if title_reason:
        return title_reason
    if requested_job_titles and not matches_requested_job_title(
        contact.get("job_title", ""), requested_job_titles
    ):
        return "title_no_match"
    if contact.get("pairing_confidence") == "invalid":
        return "email_name_mismatch"
    email = clean_text(contact.get("email", "")).lower()
    if not email:
        return "missing_email"
    if not is_valid_email(email):
        return "invalid_email"
    return None


class _TitleFromAnchor(NamedTuple):
    """Job title extracted from a mailto anchor and optional rejection metadata."""

    title: str = ""
    reject_reason: str | None = None
    raw_candidate: str = ""


def _strip_name_from_title(title: str, full_name: str) -> str:
    """Remove a captured person name from the scraped job title text."""
    cleaned_title = _clean_job_title(title)
    cleaned_name = clean_person_name(full_name)
    if not cleaned_title:
        return cleaned_title
    if not cleaned_name:
        return cleaned_title

    for separator in (" - ", " – ", " — "):
        if separator in cleaned_title:
            left, right = cleaned_title.split(separator, 1)
            if cleaned_name.lower() in clean_text(left).lower():
                cleaned_title = _clean_job_title(right)
                break

    patterns = (
        rf"^{re.escape(cleaned_name)}\s*[,;:\-–—]\s*",
        rf"^{re.escape(cleaned_name)}\s*[-–—]\s*",
        rf"\s*[,;:\-–—]\s*{re.escape(cleaned_name)}\s*$",
        rf",\s*{re.escape(cleaned_name)}\s*$",
    )
    for pattern in patterns:
        cleaned_title = re.sub(pattern, "", cleaned_title, flags=re.IGNORECASE).strip()

    return _clean_job_title(cleaned_title)


def _field_or_empty(value: object) -> str:
    """Coerce a scraped field to a string, treating nullish JS values as empty."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"undefined", "null", "none"}:
        return ""
    return text


def _normalize_contact_fields(contact: dict) -> dict:
    """Clean and normalize all string fields on a contact dict."""
    full_name = clean_person_name(_field_or_empty(contact.get("full_name", "")))
    job_title = _strip_name_from_title(
        _field_or_empty(contact.get("job_title", "")),
        full_name,
    )
    contact["full_name"] = full_name
    contact["job_title"] = job_title
    organization = clean_text(_field_or_empty(contact.get("organization", "")))
    if is_generic_page_label(organization):
        organization = ""
    contact["organization"] = organization
    contact["email"] = sanitize_scraped_email(_field_or_empty(contact.get("email", "")))
    contact["phone"] = _field_or_empty(contact.get("phone", ""))
    contact["linkedin"] = _field_or_empty(contact.get("linkedin", ""))
    contact["profile_url"] = _field_or_empty(contact.get("profile_url", ""))
    if "pairing_confidence" not in contact:
        contact["pairing_confidence"] = name_email_pairing_confidence(
            full_name, contact["email"]
        )
    contact["needs_review"] = compute_needs_review(contact)
    return contact


def _page_state_context(soup: BeautifulSoup, url: str) -> str:
    """Build a lowercase text blob from URL, headers, and a body sample."""
    parts = [url.lower()]
    if soup.title and soup.title.string:
        parts.append(clean_text(soup.title.string).lower())
    h1 = soup.find("h1")
    if h1:
        parts.append(clean_text(h1.get_text()).lower())
    meta = soup.find("meta", attrs={"property": "og:site_name"})
    if meta and meta.get("content"):
        parts.append(clean_text(meta["content"]).lower())

    body = soup.find("body")
    if body:
        body_text = clean_text(body.get_text(" ", strip=True)).lower()
        parts.append(body_text[:4000])

    return " ".join(parts)


def _extract_area_codes(text: str) -> set[str]:
    """Extract US area codes from phone-like patterns in page text."""
    codes: set[str] = set()
    for match in _PHONE_IN_TEXT.findall(text):
        digits = re.sub(r"\D", "", match)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) >= 10:
            codes.add(digits[:3])
    return codes


def _url_indicates_other_state(url: str, expected_state: str) -> bool:
    """Return True when the URL domain clearly belongs to another state."""
    host = urlparse(url).netloc.lower()
    expected_abbr = state_abbreviation(expected_state).lower()
    if f".{expected_abbr}." in host or f"k12.{expected_abbr}" in host:
        return False

    for other_state, other_abbr in STATE_ABBREVIATIONS.items():
        if other_state == expected_state:
            continue
        other_abbr = other_abbr.lower()
        if (
            f".{other_abbr}." in host
            or host.endswith(f".{other_abbr}.us")
            or f"k12.{other_abbr}" in host
        ):
            return True
    return False


def page_matches_expected_state(
    soup: BeautifulSoup,
    url: str,
    expected_state: str,
) -> bool:
    """
    Return True when a page appears to belong to the requested search state.

    Skips deep contact parsing for clearly out-of-state results (e.g. a
    Connecticut search that landed on a New York City schools page).
    """
    if not expected_state:
        return True

    if _url_indicates_other_state(url, expected_state):
        return False

    haystack = _page_state_context(soup, url)
    abbr = state_abbreviation(expected_state).lower()
    name = expected_state.lower()

    expected_markers = (
        f" {abbr} ",
        f".{abbr}.",
        f"k12.{abbr}",
        f".{abbr}.us",
        name,
        f", {abbr}",
        f" {abbr},",
    )
    has_expected = any(marker in haystack for marker in expected_markers)
    has_expected = has_expected or re.search(
        rf"\b{re.escape(abbr)}\b", haystack
    ) is not None

    has_conflict = any(
        hint in haystack for hint in get_conflict_hints_for_state(expected_state)
    )
    if not has_expected:
        for other_state, other_abbr in STATE_ABBREVIATIONS.items():
            if other_state == expected_state:
                continue
            other_abbr = other_abbr.lower()
            if (
                f"k12.{other_abbr}" in haystack
                or f".{other_abbr}.us" in haystack
                or re.search(rf"\b{re.escape(other_state.lower())}\b", haystack)
            ):
                has_conflict = True
                break

    area_codes = _extract_area_codes(haystack)
    expected_codes = get_area_codes_for_state(expected_state)
    found_expected_codes = area_codes & expected_codes
    found_other_codes: set[str] = set()
    for other_state, other_codes in STATE_AREA_CODES.items():
        if other_state == expected_state:
            continue
        found_other_codes |= area_codes & other_codes

    if found_other_codes and not found_expected_codes and not has_expected:
        return False

    if has_conflict and not has_expected:
        return False
    return True


def _is_operations_role_false_positive(title: str) -> bool:
    """Return True when a title is a facilities/operations role, not academic leadership."""
    normalized = _normalize_for_match(title)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _OPERATIONS_ROLE_PATTERNS)


def _normalize_for_match(text: str) -> str:
    """Lowercase and strip punctuation for loose substring title matching."""
    normalized = clean_text(text).lower().replace("&", "and")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _single_title_pair_matches(scraped_norm: str, req_norm: str) -> bool:
    """
    Return True when one scraped title and one requested title are a valid pair.

    Allows a requested role embedded in a longer scraped title (e.g. requested
    "Superintendent" matching scraped "Assistant Superintendent"), but rejects
    facilities/operations roles and tiny scraped fragments like "Work" that
    only match because they are substrings of a longer requested title.
    """
    if not scraped_norm or not req_norm:
        return False

    if _is_operations_role_false_positive(scraped_norm):
        return False

    if scraped_norm == req_norm:
        return True

    if req_norm in scraped_norm:
        return True

    if scraped_norm in req_norm:
        # Avoid false positives like scraped "Work" matching
        # "Work Based Learning Coordinator".
        scraped_words = scraped_norm.split()
        if len(scraped_words) >= 2:
            return True
        if len(scraped_norm) >= max(len(req_norm) * 0.6, 12):
            return True
        return False

    return False


def matches_requested_job_title(scraped_title: str, requested_titles: list[str]) -> bool:
    """
    Return True if a scraped job title loosely matches a user-requested title.

    Uses bidirectional substring matching so e.g. scraped "Assistant Superintendent"
    matches a requested title of "Superintendent", while "Principal" only matches
    when "Principal" (or a longer title containing it) is in the requested list.
    Facilities/operations roles and single-word title fragments are rejected.
    """
    scraped_norm = _normalize_for_match(scraped_title)
    if not scraped_norm or not requested_titles:
        return False

    return any(
        _single_title_pair_matches(scraped_norm, _normalize_for_match(requested))
        for requested in requested_titles
        if _normalize_for_match(requested)
    )


def matched_requested_job_title(
    scraped_title: str,
    requested_titles: list[str],
) -> str | None:
    """Return the most specific requested title that matches the scraped title."""
    matches = [
        requested
        for requested in requested_titles
        if matches_requested_job_title(scraped_title, [requested])
    ]
    if not matches:
        return None
    return max(matches, key=lambda title: len(_normalize_for_match(title)))


def _title_matches(title: str, keywords: list[str]) -> bool:
    """
    Return True if the title matches any keyword, with conditional rules.

    ``coordinator`` and ``specialist`` only match when the title also contains
    a career/education qualifier (e.g. "Career Coordinator", not "Payroll Coordinator").
    """
    normalized = _normalize_for_match(title)
    if not normalized:
        return False

    def _has_career_qualifier() -> bool:
        return any(_normalize_for_match(q) in normalized for q in _CAREER_QUALIFIERS)

    for kw in keywords:
        kw_norm = _normalize_for_match(kw)
        if kw_norm not in normalized:
            continue

        # Exact conditional term as the whole keyword — require a career qualifier
        if kw_norm in _CONDITIONAL_TITLE_TERMS:
            if _has_career_qualifier():
                return True
            continue

        # Multi-word keyword that embeds coordinator/specialist (e.g. "CTE Coordinator")
        if any(term in kw_norm for term in _CONDITIONAL_TITLE_TERMS):
            return True

        return True

    # Title contains coordinator/specialist but only generic keywords were tested
    if re.search(r"\b(coordinator|specialist)\b", normalized):
        return _has_career_qualifier()

    return False


def _looks_like_name(text: str, requested_job_titles: list[str]) -> bool:
    """Heuristic check that a string looks like a person's name, not a title."""
    text = clean_text(text)
    if not text or len(text) > 60:
        return False
    if matches_requested_job_title(text, requested_job_titles):
        return False
    if "@" in text or _PHONE_IN_TEXT.search(text):
        return False
    return is_valid_person_name(text)


def _is_valid_full_name(name: str) -> bool:
    """Validate that a scraped full_name looks like a real person."""
    return _full_name_rejection_reason(name) is None


def _is_valid_job_title(title: str) -> bool:
    """
    Reject job titles that are clearly not real titles.

    Filters navigation/link UI text, misassigned emails, program/grant names,
    and biography-length paragraphs mistaken for titles.
    """
    return _job_title_rejection_reason(title) is None


def _resolve_linkedin(contact: dict) -> None:
    """
    Populate a contact's linkedin field from the page or a Serper lookup.

    When no ``linkedin.com/in/`` link was found in the scraped HTML, searches
    for a profile URL using the contact's name and organization.
    """
    existing = clean_text(contact.get("linkedin", ""))
    if existing and re.search(r"linkedin\.com/in/", existing, re.IGNORECASE):
        contact["linkedin"] = existing
        return

    name = clean_text(contact.get("full_name", ""))
    organization = clean_text(contact.get("organization", ""))
    if not name or not organization:
        contact["linkedin"] = existing
        return

    try:
        found = find_linkedin_url(name, organization)
    except ValueError:
        found = None

    contact["linkedin"] = found or existing or ""


def _contact_is_keepable(
    contact: dict,
    requested_job_titles: list[str] | None = None,
) -> bool:
    """
    Return True only when a contact has a valid person name, valid email,
    a plausible job title, and a title that matches the user's requested list.

    Phone is optional — it is included when available but never sufficient
    on its own to keep a contact row.
    """
    return _contact_rejection_reason(contact, requested_job_titles) is None


def _organization_from_url_fallback(url: str) -> str:
    """
    Last-resort organization name derived from the page URL's domain.

    Title-cases the host slug, expands ``k12`` to ``K-12``, and strips
    leading ``isd`` prefixes (e.g. ``isd.springfield.k12.tx.us`` → ``Springfield``).
    """
    host = urlparse(url).netloc.lower().replace("www.", "")
    if not host:
        return ""
    slug = host.split(".")[0]
    slug = re.sub(r"^isd[-_]?", "", slug, flags=re.I)
    slug = re.sub(r"k12", "K-12", slug, flags=re.I)
    name = slug.replace("-", " ").replace("_", " ")
    parts: list[str] = []
    for word in name.split():
        if word.upper() == "K-12":
            parts.append("K-12")
        else:
            parts.append(word.title())
    return " ".join(parts)


def _strip_page_title_suffixes(title: str) -> str:
    """Remove common page-title suffixes such as ' - Staff Directory' or ' | Home'."""
    title = clean_text(title)
    suffix_patterns = (
        r"\s*[-|–—]\s*Staff Directory\s*$",
        r"\s*[-|–—]\s*Home\s*$",
        r"\s*[-|–—]\s*Staff\s*$",
        r"\s*[-|–—]\s*Directory\s*$",
        r"\s*[-|–—]\s*Contact(?:\s+Us)?\s*$",
        r"\s*[-|–—]\s*Official Site\s*$",
        r"\s*[-|–—]\s*Welcome\s*$",
    )
    for pattern in suffix_patterns:
        title = re.sub(pattern, "", title, flags=re.I).strip()
    return title


_GENERIC_ORG_NAMES: frozenset[str] = frozenset({
    "home", "staff", "staff directory", "directory", "contact", "welcome",
})

_ORG_H1_KEYWORDS: tuple[str, ...] = (
    "district", "school", "unified", "isd", "usd", "csd", "academy",
    "education", "public schools", "schools",
)


def _is_usable_org_name(name: str) -> bool:
    """Return True if a candidate organization name is non-empty and not generic."""
    cleaned = clean_text(name)
    return bool(cleaned) and cleaned.lower() not in _GENERIC_ORG_NAMES


def _h1_looks_like_org(text: str) -> bool:
    """Return True if an ``h1`` heading looks like a district or school name."""
    text = clean_text(text)
    if not text or not _is_usable_org_name(text):
        return False
    if _looks_like_name(text, DEFAULT_JOB_TITLE_KEYWORDS):
        return False
    if _title_matches(text, DEFAULT_JOB_TITLE_KEYWORDS):
        return False
    lower = text.lower()
    if any(kw in lower for kw in _ORG_H1_KEYWORDS):
        return True
    return bool(re.search(r"district\s+\d+", lower))


def _extract_page_organization(soup: BeautifulSoup, source_url: str) -> str:
    """
    Derive the organization name from page HTML, with URL domain as last resort.

    Fallback order:

    1. ``og:site_name`` meta tag — often the district or school brand name.
    2. ``<title>`` tag with common suffixes stripped (e.g. ``" - Staff Directory"``,
       ``" | Home"``).
    3. ``<h1>`` on the page if it looks like a district/school name.
    4. Cleaned domain name from ``source_url`` (title case, ``k12`` → ``K-12``,
       leading ``isd`` prefix stripped).

    Per-contact extraction (e.g. an accordion ``h2`` like "District 13") takes
    precedence over this page-level name when found closer to the contact entry.
    """
    meta = soup.find("meta", attrs={"property": "og:site_name"})
    if meta and meta.get("content"):
        org = clean_text(meta["content"])
        if _is_usable_org_name(org):
            return org

    if soup.title and soup.title.string:
        org = _strip_page_title_suffixes(clean_text(soup.title.string))
        if _is_usable_org_name(org):
            return org

    for h1 in soup.find_all("h1", limit=5):
        text = clean_text(h1.get_text())
        if _h1_looks_like_org(text):
            return text

    return _organization_from_url_fallback(source_url)


def _extract_organization_from_element(
    element: Tag,
    page_organization: str,
) -> str:
    """
    Find a specific organization name by walking up from a contact element.

    On pages like NYC district leadership, each accordion block has an ``h2``
    with the district name (e.g. "District 13"). Falls back to the page-level
    organization from ``_extract_page_organization`` when no local heading
    is found.
    """
    current: Tag | None = element
    for _ in range(12):
        if current is None or not isinstance(current, Tag):
            break
        if current.name in ("body", "html"):
            break

        classes = " ".join(current.get("class", [])).lower()
        is_district_block = (
            "accordionblock" in classes
            or any(kw in classes for kw in ("district", "school-block", "location-block"))
        )

        if is_district_block or _is_staff_container(current):
            for heading in current.find_all(["h1", "h2", "h3", "h4"], recursive=False):
                heading_text = clean_text(heading.get_text())
                if not heading_text:
                    continue
                # Skip headings that are clearly a person's name, not an org
                if _looks_like_name(heading_text, DEFAULT_JOB_TITLE_KEYWORDS):
                    continue
                if _title_matches(heading_text, DEFAULT_JOB_TITLE_KEYWORDS):
                    continue
                return heading_text

        current = current.parent if isinstance(current.parent, Tag) else None

    return page_organization


def _count_mailto_links(tag: Tag) -> int:
    """Count mailto anchors within a DOM subtree."""
    return len(tag.find_all("a", href=lambda h: h and h.startswith("mailto:")))


_MAILTO_SCOPE_TAGS: frozenset[str] = frozenset({
    "td", "li", "p", "dt", "dd", "span", "div", "article", "section",
})


def _mailto_entry_scope(anchor: Tag) -> Tag:
    """
    Return the tightest DOM scope for a single mailto contact.

    Prefers table cells and list items that contain exactly one mailto link.
    Stops ascending when a parent would include multiple mailto links.
    """
    best: Tag = anchor
    current: Tag | None = anchor
    for _ in range(12):
        if not isinstance(current, Tag):
            break
        parent = current.parent if isinstance(current.parent, Tag) else None
        if parent is None or parent.name in ("body", "html", "[document]"):
            break
        if _count_mailto_links(parent) > 1:
            break
        if _count_mailto_links(current) == 1 and current.name in _MAILTO_SCOPE_TAGS:
            best = current
        current = parent
        if isinstance(current, Tag) and _count_mailto_links(current) == 1:
            if current.name in _MAILTO_SCOPE_TAGS:
                best = current
    return best


def _local_text_before_anchor(anchor: Tag) -> str:
    """Collect label text from previous siblings within the anchor's parent only."""
    parent = anchor.parent if isinstance(anchor.parent, Tag) else None
    if parent is None:
        return ""

    parts: list[str] = []
    for sibling in anchor.previous_siblings:
        if isinstance(sibling, Tag):
            if sibling.name == "a" and sibling.get("href", "").startswith("mailto:"):
                return ""
            if sibling.find("a", href=lambda h: h and h.startswith("mailto:")):
                return ""
            text = sibling.get_text(" ", strip=True)
            if text:
                parts.append(text)
        else:
            text = str(sibling).strip()
            if text:
                parts.append(text)
    return clean_text(" ".join(parts))


def _finalize_mailto_contact(
    contact: dict,
    *,
    source: str,
    requested_job_titles: list[str],
) -> dict | None:
    """Normalize, validate pairing, and reject low-confidence mailto contacts."""
    _normalize_contact_fields(contact)
    reason = _contact_rejection_reason(contact, requested_job_titles)
    if reason:
        _log_rejected_contact(
            reason,
            full_name=contact.get("full_name", ""),
            job_title=contact.get("job_title", ""),
            email=contact.get("email", ""),
            source=source,
        )
        return None
    _log_chosen_job_title(contact["job_title"], source)
    return contact


def _email_from_mailto_anchor(anchor: Tag) -> str:
    """Extract email from one mailto anchor, never from sibling entries."""
    href = anchor.get("href", "")
    email = sanitize_scraped_email(href.replace("mailto:", ""))
    if is_valid_email(email):
        return email

    scope = _mailto_entry_scope(anchor)
    for link in scope.find_all("a", href=True):
        if link is not anchor:
            continue
        href = link.get("href", "")
        email = sanitize_scraped_email(href.replace("mailto:", ""))
        if is_valid_email(email):
            return email
    return ""


def _get_title_before_anchor(anchor: Tag) -> _TitleFromAnchor:
    """
    Extract a job-title label from text immediately before a mailto link.

    Prefers previous siblings in the anchor's parent (same table cell / line),
    then falls back to single-mailto scope text before the anchor name.
    """
    local = _local_text_before_anchor(anchor)
    if local:
        title = _clean_job_title(local)
        if title:
            return _TitleFromAnchor(title=title)

    scope = _mailto_entry_scope(anchor)
    if _count_mailto_links(scope) != 1:
        return _TitleFromAnchor()

    anchor_text = clean_person_name(anchor.get_text())
    if not anchor_text:
        return _TitleFromAnchor()

    scope_text = scope.get_text(" ", strip=True)
    if anchor_text not in scope_text:
        return _TitleFromAnchor()

    before = scope_text.split(anchor_text, 1)[0].strip()
    before = re.sub(r"[:.,;-]+$", "", before).strip()
    if before:
        if len(before.split()) > _CONTAINER_FALLBACK_TITLE_MAX_WORDS:
            return _TitleFromAnchor(
                reject_reason="title_too_long",
                raw_candidate=before,
            )
        title = _clean_job_title(before)
        if title:
            return _TitleFromAnchor(title=title)

    sibling = anchor.previous_sibling
    while sibling is not None:
        if isinstance(sibling, Tag):
            if sibling.name in ("strong", "b"):
                title = _clean_job_title(sibling.get_text(" ", strip=True))
                if title:
                    return _TitleFromAnchor(title=title)
            if sibling.name == "br":
                sibling = sibling.previous_sibling
                continue
            break
        sibling = sibling.previous_sibling
    return _TitleFromAnchor()


def _title_candidates_from_lines(
    lines: list[str],
    requested_job_titles: list[str],
) -> list[str]:
    """Collect job-title candidates from individual and merged consecutive lines."""
    candidates: list[str] = []
    for idx, line in enumerate(lines):
        cleaned = _clean_job_title(line)
        if matches_requested_job_title(cleaned, requested_job_titles):
            candidates.append(cleaned)
        if idx + 1 < len(lines):
            merged = _clean_job_title(f"{line} {lines[idx + 1]}")
            if matches_requested_job_title(merged, requested_job_titles):
                candidates.append(merged)
    return candidates


def _best_matching_job_title(
    candidates: list[str],
    requested_job_titles: list[str],
) -> str:
    """Return the longest valid title candidate, preferring more specific matches."""
    valid = [
        title
        for title in candidates
        if matches_requested_job_title(title, requested_job_titles)
    ]
    if not valid:
        return ""
    return max(valid, key=lambda title: len(_normalize_for_match(title)))


def _contact_entry_scope(element: Tag) -> Tag:
    """
    Return the smallest DOM node that wraps a single contact entry.

    Prefer ``p``, ``tr``, ``li``, or ``td`` — avoids pulling phone numbers
    from sibling entries in a larger accordion or directory block.
    """
    current: Tag | None = element
    for _ in range(5):
        if current is None or not isinstance(current, Tag):
            break
        if current.name in ("p", "tr", "li", "td", "dt"):
            return current
        current = current.parent if isinstance(current.parent, Tag) else None
    if isinstance(element.parent, Tag):
        return element.parent
    return element


def _first_valid_phone(text: str) -> str:
    """
    Extract the first valid 10-digit US phone number from text.

    Never concatenates multiple numbers — returns only the first match.
    """
    if not text:
        return ""
    for match in _PHONE_IN_TEXT.findall(text):
        digits = re.sub(r"\D", "", match)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) == 10:
            return normalize_phone(match)
    return ""


def _extract_phone_from_scope(scope: Tag) -> str:
    """
    Extract the first valid phone from a single contact's immediate container.

    Checks ``tel:`` links in the scope first, then plain text (including
    ``Phone:`` labels). Only searches within ``scope``, not ancestor blocks.
    """
    for link in scope.find_all("a", href=True):
        href = link.get("href", "")
        if href.startswith("tel:"):
            phone = _first_valid_phone(href.replace("tel:", ""))
            if phone:
                return phone

    block_text = scope.get_text(" ", strip=True)
    phone_label = re.search(
        r"Phone:\s*(.+?)(?:\s*(?:Fax|Email|Office)\b|$)",
        block_text,
        re.IGNORECASE,
    )
    if phone_label:
        phone = _first_valid_phone(phone_label.group(1))
        if phone:
            return phone

    return _first_valid_phone(block_text)


def _extract_phone_near_element(element: Tag) -> str:
    """Extract the first phone number from the contact's mailto entry scope."""
    return _extract_phone_from_scope(_mailto_entry_scope(element))


def _name_from_mailto_anchor(anchor: Tag) -> str:
    """Extract a person's name from the visible text of a mailto link."""
    return clean_person_name(anchor.get_text())


def _parse_mailto_entry(
    anchor: Tag,
    requested_job_titles: list[str],
    source_url: str,
    page_organization: str,
) -> dict | None:
    """
    Parse a single mailto link and its surrounding label/context into a contact.

    Designed for directory pages where each role is marked with a ``<strong>``
    label followed by a mailto link containing the person's name.
    """
    job_title_result = _get_title_before_anchor(anchor)
    full_name = _name_from_mailto_anchor(anchor)
    email = _email_from_mailto_anchor(anchor)

    if job_title_result.reject_reason:
        _log_rejected_contact(
            job_title_result.reject_reason,
            full_name=full_name,
            job_title=job_title_result.raw_candidate,
            email=email,
            source="mailto",
        )
        return None

    job_title = job_title_result.title
    if not job_title:
        _log_rejected_contact(
            "no_title_found",
            full_name=full_name,
            job_title="",
            email=email,
            source="mailto",
        )
        return None
    if not matches_requested_job_title(job_title, requested_job_titles):
        _log_rejected_contact(
            "title_no_match",
            full_name=full_name,
            job_title=job_title,
            email=email,
            source="mailto",
        )
        return None

    if not full_name:
        _log_rejected_contact(
            "missing_name",
            full_name="",
            job_title=job_title,
            email=email,
            source="mailto",
        )
        return None
    if not email:
        _log_rejected_contact(
            "missing_email",
            full_name=full_name,
            job_title=job_title,
            email="",
            source="mailto",
        )
        return None

    phone = _extract_phone_near_element(anchor)

    organization = _extract_organization_from_element(anchor, page_organization)
    container = _mailto_entry_scope(anchor)

    contact = {
        "full_name": full_name,
        "job_title": job_title,
        "organization": organization,
        "email": email,
        "phone": phone,
        "linkedin": _extract_linkedin(container),
        "profile_url": _extract_profile_url(container, source_url, full_name),
        "pairing_confidence": name_email_pairing_confidence(full_name, email),
    }
    return _finalize_mailto_contact(
        contact,
        source="mailto",
        requested_job_titles=requested_job_titles,
    )


def _is_staff_container(tag: Tag) -> bool:
    """Return True if a tag looks like a single staff-directory entry wrapper."""
    if tag.name in {"tr", "li", "article"}:
        return True
    classes = " ".join(tag.get("class", [])).lower()
    return any(kw in classes for kw in _STAFF_CLASS_KEYWORDS)


def _find_staff_container(element: Tag) -> Tag:
    """
    Walk up the DOM from an anchor/link to the nearest staff-entry container.

    Falls back to the element's immediate parent if nothing more specific is found.
    """
    current: Tag | None = element
    fallback: Tag | None = element.parent if isinstance(element.parent, Tag) else None

    for _ in range(8):
        if current is None or not isinstance(current, Tag):
            break
        if current.name in ("body", "html"):
            break
        if _is_staff_container(current):
            return current
        current = current.parent if isinstance(current.parent, Tag) else None

    return fallback or element


def _container_key(container: Tag) -> str:
    """Stable-ish key for deduplicating containers within a single page."""
    text = clean_text(container.get_text())[:200]
    return text


def _extract_email(container: Tag) -> str:
    """Extract an email from mailto: links, falling back to plain text in the container."""
    for link in container.select('a[href^="mailto:"]'):
        href = link.get("href", "")
        raw = href.replace("mailto:", "")
        email = sanitize_scraped_email(raw)
        if email:
            return email

    for match in _EMAIL_IN_TEXT.findall(container.get_text()):
        email = sanitize_scraped_email(match)
        if is_valid_email(email):
            return email

    return ""


def _extract_phone(container: Tag) -> str:
    """Extract the first valid phone from a contact's immediate container."""
    mailto = container.find("a", href=lambda h: h and h.startswith("mailto:"))
    scope = _contact_entry_scope(mailto) if mailto else container
    if scope.name not in ("p", "tr", "li", "td", "dt") and container.name in (
        "p", "tr", "li", "td",
    ):
        scope = container
    return _extract_phone_from_scope(scope)


def _extract_linkedin(container: Tag) -> str:
    """Extract a LinkedIn profile URL from anchor tags in the container."""
    for link in container.find_all("a", href=True):
        href = link["href"]
        if "linkedin.com" in href.lower():
            return clean_text(href)
    return ""


def _extract_profile_url(container: Tag, source_url: str, full_name: str) -> str:
    """
    Find an official staff profile link within the container.

    Prefers an internal link whose text matches the contact's name, then any
    link whose href suggests a profile page, then falls back to the source URL.
    """
    name_lower = full_name.lower()

    for link in container.find_all("a", href=True):
        href = link["href"]
        if href.startswith(("mailto:", "tel:", "#")):
            continue
        if "linkedin.com" in href.lower():
            continue

        link_text = clean_text(link.get_text()).lower()
        absolute = urljoin(source_url, href)

        if name_lower and name_lower in link_text:
            return absolute

        href_lower = href.lower()
        if any(kw in href_lower for kw in ("staff", "profile", "directory", "bio")):
            return absolute

    return source_url


def _lines_from_container(container: Tag) -> list[str]:
    """Split container text into cleaned, non-empty lines."""
    return [
        clean_text(line)
        for line in container.get_text("\n").split("\n")
        if clean_text(line)
    ]


def _extract_name_and_title(
    container: Tag, requested_job_titles: list[str]
) -> tuple[str, str, Tag | None]:
    """
    Extract a name/title pair from a staff-directory container.

    Strategy:
    1. Prefer mailto links: name from link text, title from preceding ``<strong>``.
    2. Scan headings and bold tags for a matching job title.
    3. Scan text lines for a matching job title, then look at the preceding line
       for a name.
    4. Fall back to the first heading/bold element that looks like a name.
    """
    job_title = ""
    full_name = ""
    matched_anchor: Tag | None = None

    # Mailto-linked entries (e.g. NYC district leadership pages)
    for anchor in container.find_all("a", href=True):
        href = anchor.get("href", "")
        if not href.startswith("mailto:"):
            continue
        candidate_title_result = _get_title_before_anchor(anchor)
        if candidate_title_result.reject_reason:
            continue
        candidate_title = candidate_title_result.title
        if not candidate_title or not matches_requested_job_title(
            candidate_title, requested_job_titles
        ):
            continue
        candidate_name = _name_from_mailto_anchor(anchor)
        if candidate_name:
            return candidate_name, candidate_title, anchor

    # Headings and bold text often carry the name or title directly
    heading_title_candidates: list[str] = []
    for tag in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b"]):
        raw = tag.get_text(" ", strip=True)
        if not raw:
            continue
        cleaned = _clean_job_title(raw)
        if matches_requested_job_title(cleaned, requested_job_titles):
            heading_title_candidates.append(cleaned)
        elif _looks_like_name(cleaned, requested_job_titles) and not full_name:
            full_name = clean_person_name(raw)

    if heading_title_candidates:
        job_title = _best_matching_job_title(
            heading_title_candidates, requested_job_titles
        )

    # Line-by-line scan: title line + name on the preceding line
    lines = _lines_from_container(container)
    if not job_title:
        line_candidates = _title_candidates_from_lines(lines, requested_job_titles)
        job_title = _best_matching_job_title(line_candidates, requested_job_titles)

    if job_title and not full_name:
        try:
            idx = next(
                i for i, line in enumerate(lines)
                if _clean_job_title(line) == job_title
            )
            if idx > 0:
                candidate = clean_person_name(lines[idx - 1])
                if _looks_like_name(candidate, requested_job_titles):
                    full_name = candidate
        except StopIteration:
            pass

    # Last resort: first name-like heading if we have a title but no name
    if job_title and not full_name:
        for tag in container.find_all(["h1", "h2", "h3", "h4", "h5", "strong", "b"]):
            text = clean_text(tag.get_text())
            if _looks_like_name(text, requested_job_titles):
                full_name = clean_person_name(text)
                break

    return full_name, job_title, matched_anchor


def _parse_container(
    container: Tag,
    requested_job_titles: list[str],
    source_url: str,
    page_organization: str,
) -> dict | None:
    """
    Parse a single staff-entry container into a contact dict.

    Returns None if the entry doesn't match a target title or lacks a valid email.
    """
    full_name, job_title, matched_anchor = _extract_name_and_title(
        container, requested_job_titles
    )

    # Must match at least one job-title keyword
    if not job_title:
        email = _extract_email(container)
        _log_rejected_contact(
            "no_title_found",
            full_name=full_name,
            job_title="",
            email=email,
            source="container",
        )
        return None

    job_title = _clean_job_title(job_title)

    if matched_anchor is not None:
        email = _email_from_mailto_anchor(matched_anchor)
    else:
        email = _extract_email(container)
    phone = _extract_phone(container)

    if not email or not is_valid_email(email):
        _log_rejected_contact(
            "missing_email" if not email else "invalid_email",
            full_name=full_name,
            job_title=job_title,
            email=email,
            source="container",
        )
        return None

    organization = _extract_organization_from_element(container, page_organization)

    contact = {
        "full_name": full_name,
        "job_title": job_title,
        "organization": organization,
        "email": email,
        "phone": phone,
        "linkedin": _extract_linkedin(container),
        "profile_url": _extract_profile_url(container, source_url, full_name),
        "pairing_confidence": name_email_pairing_confidence(full_name, email),
    }
    return _finalize_mailto_contact(
        contact,
        source="container",
        requested_job_titles=requested_job_titles,
    )


def _collect_containers(soup: BeautifulSoup) -> list[Tag]:
    """
    Gather candidate staff-entry containers from the page.

    Starts from every mailto:/tel: link and walks up to a staff wrapper.
    Also scans elements whose class names suggest staff directory cards.
    """
    seen_keys: set[str] = set()
    containers: list[Tag] = []

    def _add(container: Tag) -> None:
        key = _container_key(container)
        if key and key not in seen_keys:
            seen_keys.add(key)
            containers.append(container)

    # Anchor-based discovery: mailto and tel links almost always mark a person
    for link in soup.select('a[href^="mailto:"], a[href^="tel:"]'):
        if isinstance(link, Tag):
            _add(_find_staff_container(link))

    # Class-based discovery for cards that may not use mailto:/tel: yet
    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        classes = " ".join(tag.get("class", [])).lower()
        if any(kw in classes for kw in _STAFF_CLASS_KEYWORDS):
            _add(tag)

    # Table rows in staff directories
    for row in soup.find_all("tr"):
        if isinstance(row, Tag) and row.find("a", href=re.compile(r"^mailto:|^tel:")):
            _add(row)

    return containers


class ContactScraper:
    """
    Scrapes school district / education websites for staff contact info.

    Intended to be used as an async context manager so a single browser
    instance can be reused across many `fetch_page` calls:

        async with ContactScraper() as scraper:
            html = await scraper.fetch_page("https://example-k12.org/staff")
            contacts = await scraper.extract_contacts(html, "https://example-k12.org/staff")
    """

    def __init__(self, headless: bool = True, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> None:
        """
        Args:
            headless: Whether to launch the browser in headless mode.
            timeout_ms: Navigation timeout in milliseconds for each page load.
        """
        self.headless = headless
        self.timeout_ms = timeout_ms

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> "ContactScraper":
        """Launch the browser and open a shared context on entering the block."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
    headless=self.headless,
    args=[
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-gpu",
        "--single-process",
        "--no-zygote",
    ],
)
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the context, browser, and Playwright driver on exiting the block."""
        await self.close()

    async def close(self) -> None:
        """Tear down the browser context, browser, and Playwright driver, if open."""
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def fetch_page(self, url: str) -> str:
        """
        Load a page in the shared browser context and return its HTML.

        Args:
            url: The URL to load.

        Returns:
            The page's fully rendered HTML content, or an empty string if
            the page failed to load (e.g. timeout, DNS error, bad response).
        """
        if self._context is None:
            raise RuntimeError(
                "ContactScraper must be used as an async context manager "
                "(e.g. `async with ContactScraper() as scraper:`) before calling fetch_page."
            )

        for attempt in range(2):
            try:
                page = await self._context.new_page()
            except Exception as exc:
                print(f"[fetch_page] browser/context dead, relaunching: {exc!r}")
                await self._relaunch()
                continue

            try:
                await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                html = await page.content()
                return html
            except Exception as exc:
                print(f"[fetch_page] failed to load {url}: {exc!r}")
                return ""
            finally:
                try:
                    await page.close()
                except Exception:
                    pass

        return ""

    async def _relaunch(self) -> None:
        """Tear down and recreate the browser + context after a crash."""
        try:
            if self._context is not None:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:
            pass

        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
                "--single-process",
                "--no-zygote",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )

    async def extract_contacts(
        self,
        html: str,
        source_url: str,
        requested_job_titles: list[str] | None = None,
        expected_state: str | None = None,
    ) -> list[dict]:
        """
        Parse contact information out of a page's HTML.

        Looks for staff directory patterns: name/title pairs, mailto: emails,
        tel: phone numbers, and LinkedIn profile links. Only contacts whose
        scraped job title matches one of ``requested_job_titles`` are kept.

        Args:
            html: The page's HTML content, as returned by `fetch_page`.
            source_url: The URL the HTML was fetched from (used for organization
                name derivation and profile URL resolution).
            requested_job_titles: Job titles selected by the user. Contacts are
                filtered to those whose scraped title loosely matches one of
                these strings (substring match in either direction).
            expected_state: When set, skip parsing pages that clearly belong to
                a different state (checked via URL, title, and header text).

        Returns:
            A list of contact dicts with keys:
            full_name, job_title, organization, email, phone, linkedin,
            profile_url. A contact must have both a valid name and a valid
            email; phone is included when available but is not sufficient alone.
            LinkedIn is taken from the page when present, otherwise resolved
            via a Serper search on name and organization.
        """
        if not html or not html.strip():
            return []

        if _is_excluded_domain(source_url):
            return []

        if not requested_job_titles:
            return []

        soup = BeautifulSoup(html, "html.parser")

        if expected_state and not page_matches_expected_state(
            soup, source_url, expected_state
        ):
            return []

        page_organization = _extract_page_organization(soup, source_url)

        contacts: list[dict] = []
        seen: set[str] = set()

        def _add_contact(contact: dict | None) -> None:
            if contact is None:
                return
            if not _contact_is_keepable(contact, requested_job_titles):
                return
            _resolve_linkedin(contact)
            dedup = clean_text(contact.get("email", "")).lower()
            if dedup in seen:
                return
            seen.add(dedup)
            contacts.append(contact)

        # Primary: one contact per mailto link (handles multi-person paragraphs)
        for anchor in soup.find_all("a", href=True):
            if not anchor.get("href", "").startswith("mailto:"):
                continue
            _add_contact(
                _parse_mailto_entry(
                    anchor, requested_job_titles, source_url, page_organization
                )
            )

        # Fallback: container-based parsing for pages without mailto structure
        mailto_parents = {
            id(anchor.parent)
            for anchor in soup.find_all("a", href=True)
            if anchor.get("href", "").startswith("mailto:") and isinstance(anchor.parent, Tag)
        }
        for container in _collect_containers(soup):
            if id(container) in mailto_parents:
                continue
            _add_contact(
                _parse_container(
                    container, requested_job_titles, source_url, page_organization
                )
            )

        return contacts
