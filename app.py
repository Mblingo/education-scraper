"""
K-12 Education Contact Scraper — Streamlit application entry point.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import difflib
import subprocess
import sys
import os
import tempfile
import threading
import time

import pandas as pd
import streamlit as st

from exporter import EXPORT_COLUMNS, export_to_excel
from scraper import ContactScraper, matched_requested_job_title
from districts import get_districts_for_state

# One-time Playwright browser install for Streamlit Cloud
if not os.path.exists("/home/appuser/.cache/ms-playwright"):
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
from search import build_search_query_plan, search_web, DEFAULT_RESULTS_PER_QUERY
from utils import dedupe_key

# How often the main thread refreshes the UI while scraping (seconds)
_POLL_INTERVAL_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

US_STATES: list[str] = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
]

KNOWN_JOB_TITLES: list[str] = [
    "Superintendent",
    "Deputy Superintendent",
    "Assistant Superintendent",
    "Associate Superintendent",
    "Chief Academic Officer",
    "Director of Career & Technical Education",
    "Executive Director of Career & Technical Education",
    "Director of College & Career Readiness",
    "Executive Director of College & Career Readiness",
    "Director of Career Pathways",
    "Director of Workforce Development",
    "Director of Employer Partnerships",
    "Director of Secondary Education",
    "Director of Experiential Learning",
    "Work-Based Learning Coordinator",
    "Work-Based Learning Specialist",
    "Career Readiness Coordinator",
    "Career Pathways Coordinator",
    "CTE Coordinator",
    "Principal",
    "Principal of a Career Academy",
]

CUSTOM_TITLE_OPTION = "+ Type a custom title"

_DISPLAY_FIELDS: list[tuple[str, str]] = [
    ("full_name", "Full Name"),
    ("job_title", "Job Title"),
    ("organization", "Organization"),
    ("state", "State"),
    ("email", "Email"),
    ("phone", "Phone"),
    ("linkedin", "LinkedIn"),
    ("profile_url", "Official Profile URL"),
    ("needs_review", "Needs Review"),
]


# ---------------------------------------------------------------------------
# Thread-safe scrape state (shared with background worker — NOT session_state)
# ---------------------------------------------------------------------------

class ScrapeRunState:
    """
    Thread-safe holder for scrape progress.

    The background collection thread reads/writes this object only.
    The main Streamlit thread snapshots it for UI updates.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contacts: list[dict] = []
        self.seen_keys: set[str] = set()
        self.stop_requested = False
        self.is_running = False
        self.status_message = "Ready."

    def reset_for_run(self) -> None:
        """Clear results and mark a new scrape as running."""
        with self._lock:
            self.contacts = []
            self.seen_keys = set()
            self.stop_requested = False
            self.is_running = True
            self.status_message = "Starting scraper…"

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True
            self.status_message = "Stop requested — finishing current page…"

    def should_stop(self) -> bool:
        with self._lock:
            return self.stop_requested

    def contact_count(self) -> int:
        with self._lock:
            return len(self.contacts)

    def set_status(self, message: str) -> None:
        with self._lock:
            self.status_message = message

    def try_add_contact(self, contact: dict, key: str, max_contacts: int) -> bool:
        """
        Add a contact if it is new and under the limit.

        Returns True when the max contact count has been reached.
        """
        with self._lock:
            if key in self.seen_keys:
                return len(self.contacts) >= max_contacts
            self.seen_keys.add(key)
            self.contacts.append(contact)
            return len(self.contacts) >= max_contacts

    def finish(self, max_contacts: int, *, queries_exhausted: bool = False) -> None:
        with self._lock:
            self.is_running = False
            count = len(self.contacts)
            if self.stop_requested:
                self.status_message = f"Stopped — collected {count} contacts."
            elif count >= max_contacts:
                self.status_message = f"Done — reached limit of {max_contacts} contacts."
            elif queries_exhausted:
                self.status_message = (
                    f"Done — collected {count} contacts "
                    f"(all search queries exhausted)."
                )
            else:
                self.status_message = f"Done — collected {count} contacts."

    def set_error(self, exc: Exception) -> None:
        with self._lock:
            self.is_running = False
            self.status_message = f"Error: {exc}"

    def mark_thread_exited(self) -> None:
        """Called by the main thread if the worker dies unexpectedly."""
        with self._lock:
            if self.is_running:
                self.is_running = False
                self.status_message = (
                    f"Scraper thread exited — collected {len(self.contacts)} contacts."
                )

    def snapshot(self) -> tuple[list[dict], str, bool]:
        """Return a copy of contacts, status message, and is_running flag."""
        with self._lock:
            return list(self.contacts), self.status_message, self.is_running


# ---------------------------------------------------------------------------
# Session state helpers (main thread only)
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialize session-state keys used by the scraper UI."""
    if "scrape_run" not in st.session_state:
        st.session_state.scrape_run = ScrapeRunState()
    if "scraper_thread" not in st.session_state:
        st.session_state.scraper_thread = None
    if "excel_bytes" not in st.session_state:
        st.session_state.excel_bytes = None
    if "custom_job_titles" not in st.session_state:
        st.session_state.custom_job_titles = []


def _fuzzy_title_suggestion(title: str) -> str | None:
    """Return a close KNOWN_JOB_TITLES match for a custom title, if any."""
    cleaned = title.strip()
    if not cleaned:
        return None
    if cleaned in KNOWN_JOB_TITLES:
        return None
    matches = difflib.get_close_matches(
        cleaned, KNOWN_JOB_TITLES, n=1, cutoff=0.8
    )
    return matches[0] if matches else None


def _build_job_titles(
    selected_known: list[str],
    custom_titles: list[str],
) -> list[str]:
    """Combine known multiselect picks and custom typed titles (deduped, ordered)."""
    combined: list[str] = []
    seen: set[str] = set()
    for title in selected_known + custom_titles:
        cleaned = title.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(cleaned)
    return combined


def _add_custom_title() -> None:
    """Append the drafted custom title and clear the input (runs before widgets render)."""
    draft = st.session_state.get("custom_title_draft", "")
    new_title = draft.strip()
    if not new_title:
        return

    existing_lower = {t.lower() for t in st.session_state.custom_job_titles}
    if new_title.lower() not in existing_lower:
        st.session_state.custom_job_titles.append(new_title)
    st.session_state.custom_title_draft = ""


def _render_custom_title_input(disabled: bool) -> None:
    """UI for adding and managing custom job titles."""
    st.markdown("**Custom job titles**")

    input_col, add_col = st.columns([4, 1])
    with input_col:
        draft = st.text_input(
            "Custom job title",
            placeholder="e.g. Deputy Director of Adult Education",
            disabled=disabled,
            key="custom_title_draft",
            label_visibility="collapsed",
        )

    suggestion = _fuzzy_title_suggestion(draft)
    if suggestion:
        st.caption(f"Did you mean: **{suggestion}**? You can still add your entry as typed.")

    with add_col:
        st.button(
            "Add",
            disabled=disabled or not draft.strip(),
            use_container_width=True,
            key="add_custom_title_btn",
            on_click=_add_custom_title,
        )

    if st.session_state.custom_job_titles:
        for idx, title in enumerate(st.session_state.custom_job_titles):
            row_col, remove_col = st.columns([5, 1])
            with row_col:
                st.text(f"• {title}")
            with remove_col:
                if st.button(
                    "Remove",
                    key=f"remove_custom_title_{idx}",
                    disabled=disabled,
                    use_container_width=True,
                ):
                    st.session_state.custom_job_titles.pop(idx)
                    st.rerun()
    else:
        st.caption("No custom titles added yet.")


def _get_scrape_run() -> ScrapeRunState:
    return st.session_state.scrape_run


def _contacts_to_dataframe(contacts: list[dict]) -> pd.DataFrame:
    """Convert internal contact dicts to a display DataFrame."""
    rows = [
        {
            col: (
                str(bool(contact.get(field, False)))
                if field == "needs_review"
                else str(contact.get(field, "") or "")
            )
            for field, col in _DISPLAY_FIELDS
        }
        for contact in contacts
    ]
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS)


def _build_excel_bytes(contacts: list[dict]) -> bytes:
    """Write contacts to a temp .xlsx file and return the raw bytes."""
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_excel(contacts, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _render_results(
    placeholder: st.delta_generator.DeltaGenerator,
    scrape_run: ScrapeRunState,
) -> None:
    """Render the contact count, table, and status into a placeholder container."""
    contacts, status_message, _ = scrape_run.snapshot()
    count = len(contacts)

    with placeholder.container():
        st.metric("Contacts collected", count, border=True)

        if contacts:
            st.dataframe(
                _contacts_to_dataframe(contacts),
                use_container_width=True,
            )
        else:
            st.info("No contacts yet. Results will appear here as they are found.")

        st.caption(status_message)


# ---------------------------------------------------------------------------
# Async scraping (runs in a background thread — uses ScrapeRunState only)
# ---------------------------------------------------------------------------

def _print_per_title_counts(per_title_counts: dict[str, int], total: int) -> None:
    """Print a debug breakdown of contacts collected per requested job title."""
    print("\n=== Contacts per requested job title ===")
    for title, count in per_title_counts.items():
        print(f"  {title}: {count}")
    print(f"  Total: {total}")
    print("========================================\n")


async def _collect_contacts(
    scrape_run: ScrapeRunState,
    states: list[str],
    job_titles: list[str],
    max_contacts: int,
) -> None:
    """
    Search the web and scrape result URLs for matching contacts.

    Iterates every (job_title, state, query variation) in the search plan
    until ``max_contacts`` is reached or all queries are exhausted.
    """
    query_plan = build_search_query_plan(states, job_titles)
    visited_urls: set[str] = set()
    per_title_counts: dict[str, int] = {title: 0 for title in job_titles}
    queries_run = 0
    district_query_count = sum(
        len(get_districts_for_state(state)) * len(job_titles) * 2
        for state in states
    )

    scrape_run.set_status(
        f"Starting — {len(query_plan)} search queries across "
        f"{len(job_titles)} job title(s) "
        f"({district_query_count} district-targeted, "
        f"{DEFAULT_RESULTS_PER_QUERY} results per query)…"
    )

    async with ContactScraper() as scraper:
        for query, state, job_title in query_plan:
            if scrape_run.should_stop():
                break
            if scrape_run.contact_count() >= max_contacts:
                break

            queries_run += 1
            scrape_run.set_status(
                f"Searching ({scrape_run.contact_count()}/{max_contacts} contacts, "
                f"title: {job_title}): {query}"
            )

            urls = search_web(query)

            for url in urls:
                if scrape_run.should_stop():
                    break
                if scrape_run.contact_count() >= max_contacts:
                    break
                if url in visited_urls:
                    continue
                visited_urls.add(url)

                scrape_run.set_status(f"Scraping: {url}")

                html = await scraper.fetch_page(url)
                if not html:
                    continue

                extracted = await scraper.extract_contacts(
                    html,
                    url,
                    requested_job_titles=job_titles,
                    expected_state=state,
                )

                for contact in extracted:
                    contact["state"] = state
                    key = dedupe_key(
                        contact.get("full_name", ""),
                        contact.get("organization", ""),
                    )
                    before_count = scrape_run.contact_count()
                    at_limit = scrape_run.try_add_contact(contact, key, max_contacts)
                    if scrape_run.contact_count() > before_count:
                        matched = matched_requested_job_title(
                            contact.get("job_title", ""),
                            job_titles,
                        )
                        if matched:
                            per_title_counts[matched] = per_title_counts.get(matched, 0) + 1
                    if at_limit:
                        break

    _print_per_title_counts(per_title_counts, scrape_run.contact_count())
    scrape_run.finish(
        max_contacts,
        queries_exhausted=queries_run >= len(query_plan),
    )


def _run_collection_thread(
    scrape_run: ScrapeRunState,
    states: list[str],
    job_titles: list[str],
    max_contacts: int,
) -> None:
    """Entry point for the background thread that runs the async scraper."""
    try:
        asyncio.run(_collect_contacts(scrape_run, states, job_titles, max_contacts))
    except Exception as exc:
        scrape_run.set_error(exc)


def _start_scraping(
    scrape_run: ScrapeRunState,
    states: list[str],
    job_titles: list[str],
    max_contacts: int,
) -> None:
    """Reset state and launch the scraper in a background thread."""
    scrape_run.reset_for_run()
    st.session_state.excel_bytes = None

    thread = threading.Thread(
        target=_run_collection_thread,
        args=(scrape_run, states, job_titles, max_contacts),
        daemon=True,
    )
    thread.start()
    st.session_state.scraper_thread = thread


def _stop_scraping(scrape_run: ScrapeRunState) -> None:
    """Signal the background scraper to stop after the current URL."""
    scrape_run.request_stop()


def _poll_while_running(
    results_placeholder: st.delta_generator.DeltaGenerator,
    scrape_run: ScrapeRunState,
) -> None:
    """
    Keep the main thread alive, refreshing the results UI while scraping runs.

    Reads from ``scrape_run`` (not ``st.session_state``) and calls
    ``st.rerun()`` so button clicks (e.g. Stop) are processed between refreshes.
    """
    thread: threading.Thread | None = st.session_state.scraper_thread

    if thread is not None and not thread.is_alive():
        _, _, is_running = scrape_run.snapshot()
        if is_running:
            scrape_run.mark_thread_exited()

    _render_results(results_placeholder, scrape_run)

    _, _, is_running = scrape_run.snapshot()
    if is_running:
        time.sleep(_POLL_INTERVAL_SECONDS)
        st.rerun()


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="K-12 Education Contact Scraper",
    page_icon="🎓",
    layout="wide",
)

_init_session_state()
scrape_run = _get_scrape_run()
contacts, status_message, is_running = scrape_run.snapshot()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("K-12 Education Contact Scraper")
st.markdown(
    "Find K-12 education professionals by state and job title, "
    "then export results to Excel."
)

st.divider()

# ---------------------------------------------------------------------------
# Search inputs
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    selected_states = st.multiselect(
        "State",
        options=US_STATES,
        placeholder="Select one or more states…",
        help="Choose the state(s) to search within.",
        disabled=is_running,
    )

with col2:
    max_contacts = st.number_input(
        "Maximum Contacts",
        min_value=1,
        max_value=500,
        value=50,
        step=10,
        help="Stop collecting once this many contacts are found.",
        disabled=is_running,
    )

selected_multiselect = st.multiselect(
    "Job Title",
    options=KNOWN_JOB_TITLES + [CUSTOM_TITLE_OPTION],
    default=[],
    help="Select known titles and/or choose '+ Type a custom title' to add your own.",
    disabled=is_running,
)

selected_known_titles = [
    title for title in selected_multiselect if title != CUSTOM_TITLE_OPTION
]

if CUSTOM_TITLE_OPTION in selected_multiselect:
    _render_custom_title_input(disabled=is_running)

job_titles = _build_job_titles(
    selected_known_titles,
    st.session_state.custom_job_titles,
)

if job_titles:
    with st.expander("Final job titles (preview)", expanded=True):
        st.write(job_titles)
        st.caption(
            f"{len(job_titles)} title(s) ready — known and custom entries combined."
        )
else:
    st.caption("Select at least one known title or add a custom title.")

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------

st.divider()

btn_col1, btn_col2, btn_col3 = st.columns(3)

inputs_valid = bool(selected_states and job_titles)

with btn_col1:
    if st.button(
        "Start Scraping",
        type="primary",
        disabled=is_running or not inputs_valid,
        use_container_width=True,
    ):
        _start_scraping(scrape_run, selected_states, job_titles, max_contacts)

with btn_col2:
    if st.button(
        "Stop",
        disabled=not is_running,
        use_container_width=True,
    ):
        _stop_scraping(scrape_run)

with btn_col3:
    if st.button(
        "Export to Excel",
        disabled=not contacts or is_running,
        use_container_width=True,
    ):
        st.session_state.excel_bytes = _build_excel_bytes(contacts)

if not inputs_valid and not is_running:
    st.info("Select at least one state and one job title to enable scraping.")

# ---------------------------------------------------------------------------
# Download (shown after Export to Excel is clicked)
# ---------------------------------------------------------------------------

if st.session_state.excel_bytes:
    st.download_button(
        label="Download Excel file",
        data=st.session_state.excel_bytes,
        file_name="k12_contacts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Live results — placeholder refreshed by the main-thread poll loop
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Results")

results_placeholder = st.empty()
_poll_while_running(results_placeholder, scrape_run)
