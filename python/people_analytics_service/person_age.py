"""Date-of-birth parsing and real-time age calculation."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

_DMY_RE = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")


def parse_date_of_birth(raw: str | date | None) -> date | None:
    """Parse DD/MM/YYYY, D-M-YYYY, or ISO YYYY-MM-DD."""
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    text = str(raw).strip()
    if not text:
        return None

    m = _DMY_RE.match(text)
    if m:
        day, month, year = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        try:
            return date(year, month, day)
        except ValueError:
            return None

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def calculate_age(dob: date | None, *, on: date | None = None) -> int | None:
    """Return completed years of age at ``on`` (default: today UTC)."""
    if dob is None:
        return None
    today = on or datetime.now(timezone.utc).date()
    years = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    if years < 0 or years > 150:
        return None
    return years


def effective_age(
    *,
    date_of_birth: date | str | None,
    stored_age: int | None = None,
    on: date | None = None,
) -> int | None:
    """Prefer age computed from DOB; fall back to legacy stored age."""
    dob = parse_date_of_birth(date_of_birth) if date_of_birth else None
    if dob is not None:
        return calculate_age(dob, on=on)
    return stored_age
