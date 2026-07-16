"""
Excel export module for collected K-12 education contacts.

Writes deduplicated contact data to an .xlsx file using pandas and openpyxl,
with auto-sized columns for readability.
"""

from __future__ import annotations

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

# Column order for the exported spreadsheet
EXPORT_COLUMNS: list[str] = [
    "Full Name",
    "Job Title",
    "Organization",
    "State",
    "Email",
    "Phone",
    "LinkedIn",
    "Official Profile URL",
    "Needs Review",
]

# Maps internal contact-dict keys to export column headers
_FIELD_TO_COLUMN: dict[str, str] = {
    "full_name": "Full Name",
    "job_title": "Job Title",
    "organization": "Organization",
    "state": "State",
    "email": "Email",
    "phone": "Phone",
    "linkedin": "LinkedIn",
    "profile_url": "Official Profile URL",
    "needs_review": "Needs Review",
}


def _dedupe_by_email(contacts: list[dict]) -> list[dict]:
    """
    Remove duplicate contacts, keeping the first occurrence per email.

    Deduplication is case-insensitive on the email field. Contacts without
    an email are always kept (they cannot be deduplicated by email).

    Args:
        contacts: Raw list of contact dictionaries.

    Returns:
        Deduplicated list preserving original order.
    """
    seen_emails: set[str] = set()
    deduped: list[dict] = []

    for contact in contacts:
        email = (contact.get("email") or "").strip().lower()
        if email:
            if email in seen_emails:
                continue
            seen_emails.add(email)
        deduped.append(contact)

    return deduped


def _contact_to_row(contact: dict) -> dict[str, str]:
    """Map an internal contact dict to a row keyed by export column names."""
    row: dict[str, str] = {}
    for field, column in _FIELD_TO_COLUMN.items():
        value = contact.get(field, "")
        if field == "needs_review":
            row[column] = str(bool(value))
            continue
        row[column] = str(value).strip() if value is not None else ""
    return row


def _autosize_columns(worksheet: Worksheet, df: pd.DataFrame) -> None:
    """
    Set each column width based on the longest cell value in that column.

    Caps width at 60 characters to avoid excessively wide columns from URLs.
    """
    for idx, column in enumerate(df.columns, start=1):
        series = df[column].astype(str)
        max_length = max(series.map(len).max(), len(column))
        adjusted_width = min(max_length + 2, 60)
        worksheet.column_dimensions[get_column_letter(idx)].width = adjusted_width


def export_to_excel(contacts: list[dict], filepath: str) -> None:
    """
    Export a list of contacts to an Excel (.xlsx) file.

    Contacts are deduplicated by email (case-insensitive) before writing.
    Columns are auto-sized for readability.

    Args:
        contacts: List of contact dictionaries. Expected keys include
            full_name, job_title, organization, state, email, phone,
            linkedin, and profile_url (all optional).
        filepath: Destination path for the .xlsx file.

    Returns:
        None
    """
    deduped = _dedupe_by_email(contacts)
    rows = [_contact_to_row(contact) for contact in deduped]
    df = pd.DataFrame(rows, columns=EXPORT_COLUMNS)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Contacts")
        worksheet = writer.sheets["Contacts"]
        _autosize_columns(worksheet, df)
