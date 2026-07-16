"""
End-to-end debug script for the K-12 education contact scraper.

Runs a single hardcoded search → fetch → extract pipeline and prints
results at each stage. Not tied to Streamlit.

Usage (from the education_scraper/ directory):
    python debug_test.py
"""

from __future__ import annotations

import asyncio
import json

from scraper import ContactScraper
from search import build_search_queries, search_web

# Hardcoded test inputs
STATES = ["New York"]
JOB_TITLES = ["Superintendent"]


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


async def _fetch_page(url: str) -> str:
    """Fetch a single page and return its HTML."""
    async with ContactScraper() as scraper:
        return await scraper.fetch_page(url)


async def _extract_contacts(html: str, url: str) -> list[dict]:
    """Extract contacts from HTML (parsing only — no browser launch needed)."""
    scraper = ContactScraper()
    return await scraper.extract_contacts(html, url, requested_job_titles=JOB_TITLES)


def main() -> None:
    # ------------------------------------------------------------------
    # Stage 1: Build search queries
    # ------------------------------------------------------------------
    _section("Stage 1: build_search_queries")
    queries = build_search_queries(STATES, JOB_TITLES)
    print(f"States:     {STATES}")
    print(f"Job titles: {JOB_TITLES}")
    print(f"Queries generated ({len(queries)}):")
    for i, query in enumerate(queries, start=1):
        print(f"  {i}. {query}")

    if not queries:
        print("No queries generated — stopping.")
        return

    first_query = queries[0]

    # ------------------------------------------------------------------
    # Stage 2: Search the web
    # ------------------------------------------------------------------
    _section("Stage 2: search_web")
    print(f"Query: {first_query}")
    urls = search_web(first_query)
    print(f"URLs returned: {len(urls)}")
    if urls:
        print("First 5 URLs:")
        for i, url in enumerate(urls[:5], start=1):
            print(f"  {i}. {url}")
    else:
        print("No URLs returned — stopping.")
        return

    first_url = urls[0]

    # ------------------------------------------------------------------
    # Stage 3: Fetch page HTML
    # ------------------------------------------------------------------
    _section("Stage 3: ContactScraper.fetch_page")
    print(f"URL: {first_url}")
    html = asyncio.run(_fetch_page(first_url))
    print(f"HTML length: {len(html)} characters")
    if html:
        print("First 500 characters of HTML:")
        print("-" * 40)
        print(html[:500])
        print("-" * 40)
    else:
        print("Empty HTML returned — page may have failed to load.")
        return

    # ------------------------------------------------------------------
    # Stage 4: Extract contacts
    # ------------------------------------------------------------------
    _section("Stage 4: ContactScraper.extract_contacts")
    contacts = asyncio.run(_extract_contacts(html, first_url))
    print(f"Contacts extracted: {len(contacts)}")
    if contacts:
        print(json.dumps(contacts, indent=2))
    else:
        print("No contacts found on this page.")

    _section("Done")


if __name__ == "__main__":
    main()
