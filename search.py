"""
Web search module for finding K-12 education professional pages.

Builds targeted search queries and runs them against the Serper.dev API
to collect candidate result URLs. Synchronous for now — can be wrapped
with asyncio/threads later if needed.
"""

from __future__ import annotations

import os
import re

import requests
from dotenv import load_dotenv

from districts import get_districts_for_state, state_abbreviation

load_dotenv()

_SERPER_SEARCH_URL = "https://google.serper.dev/search"

# Serper ``num`` parameter — request more organic results per API call (max ~30)
DEFAULT_RESULTS_PER_QUERY = 25

# LinkedIn profile lookups only need a handful of top search hits
_LINKEDIN_LOOKUP_RESULTS = 5
_LINKEDIN_IN_PATTERN = re.compile(r"linkedin\.com/in/", re.IGNORECASE)

# Rotating phrasing templates appended after "{job_title} {state}"
_QUERY_SUFFIX_TEMPLATES: tuple[str, ...] = (
    "school district staff directory",
    "public schools contact",
    "school district directory",
    "K-12 staff directory",
)


# Per-district query template: "{district} {ST} school district {job_title} staff directory"
_DISTRICT_QUERY_SUFFIX = "staff directory"


def build_query_variations(job_title: str, state: str) -> list[str]:
    """
    Build multiple search query phrasings for a single job title and state.

    Rotates suffix patterns to surface different result sets from Serper.
    """
    base = f"{job_title} {state}"
    return [f"{base} {suffix}" for suffix in _QUERY_SUFFIX_TEMPLATES]


def build_district_query(job_title: str, state: str, district: str) -> str:
    """
    Build a targeted search query for one district and job title.

    Example: ``Brookfield CT school district Superintendent staff directory``
    """
    abbr = state_abbreviation(state)
    return (
        f"{district} {abbr} school district {job_title} {_DISTRICT_QUERY_SUFFIX}"
    )


def build_district_query_variations(
    job_title: str,
    state: str,
    district: str,
) -> list[str]:
    """Build district-targeted query phrasings for a single district."""
    abbr = state_abbreviation(state)
    base = f"{district} {abbr} school district {job_title}"
    return [
        f"{base} {_DISTRICT_QUERY_SUFFIX}",
        f"{base} contact",
    ]


def format_search_query(job_title: str, state: str) -> str:
    """
    Build the primary search query string for a job title and state.

    Returns the first (most specific) variation from ``build_query_variations``.
    """
    return build_query_variations(job_title, state)[0]


def build_search_queries(states: list[str], job_titles: list[str]) -> list[str]:
    """
    Build all search query strings for every state, job title, and phrasing variant.

    Args:
        states: List of US state names to search within.
        job_titles: List of job titles to search for.

    Returns:
        A flat list of query strings, including several variations per
        (job_title, state) pair.
    """
    queries: list[str] = []
    for job_title in job_titles:
        for state in states:
            queries.extend(build_query_variations(job_title, state))
    return queries


def build_search_query_plan(
    states: list[str],
    job_titles: list[str],
) -> list[tuple[str, str, str]]:
    """
    Build an ordered search plan of ``(query, state, job_title)`` triples.

    Every combination of job title, state, and query phrasing variation is
    included. Callers should iterate the full plan, stopping only when
    ``max_contacts`` unique contacts have been collected or every variation
    has been exhausted — not when a single query returns few or no URLs.
    """
    plan: list[tuple[str, str, str]] = []
    for job_title in job_titles:
        for state in states:
            for query in build_query_variations(job_title, state):
                plan.append((query, state, job_title))

            for district in get_districts_for_state(state):
                for query in build_district_query_variations(
                    job_title, state, district
                ):
                    plan.append((query, state, job_title))
    return plan


def search_web(query: str, max_results: int = DEFAULT_RESULTS_PER_QUERY) -> list[str]:
    """
    Run a single search query via the Serper.dev API and return result URLs.

    Args:
        query: The search query string.
        max_results: Number of organic results to request (Serper ``num`` param;
            defaults to 25).

    Returns:
        A list of result URLs from the ``organic`` results, in ranked order.
        Returns an empty list if the request fails or no results are found.

    Raises:
        ValueError: If the ``SERPER_API_KEY`` environment variable is not set.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError(
            "SERPER_API_KEY is not set. Add it to your .env file "
            "(see .env.example) and restart the application."
        )

    # Serper accepts num up to ~100; cap at 30 for sensible page size
    num = min(max(max_results, 1), 30)

    try:
        response = requests.post(
            _SERPER_SEARCH_URL,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    data = response.json()
    urls: list[str] = []
    seen: set[str] = set()

    for result in data.get("organic", []):
        link = result.get("link", "")
        if link and link not in seen:
            seen.add(link)
            urls.append(link)
        if len(urls) >= num:
            break

    return urls


def find_linkedin_url(name: str, organization: str) -> str | None:
    """
    Search for a LinkedIn profile URL for a person at an organization.

    Builds a Serper query like ``"Jane Doe" "Springfield ISD" site:linkedin.com/in``
    and returns the first result URL containing ``linkedin.com/in/``. Does not
    fetch or scrape the LinkedIn page itself.

    Args:
        name: The contact's full name.
        organization: The contact's organization or school district.

    Returns:
        A LinkedIn profile URL, or None if no matching result appears in the
        top few organic listings (or if the search request fails).
    """
    name = name.strip()
    organization = organization.strip()
    if not name or not organization:
        return None

    query = f'"{name}" "{organization}" site:linkedin.com/in'

    try:
        urls = search_web(query, max_results=_LINKEDIN_LOOKUP_RESULTS)
    except ValueError:
        return None

    for url in urls:
        if _LINKEDIN_IN_PATTERN.search(url):
            return url

    return None
