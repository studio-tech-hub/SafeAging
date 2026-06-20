"""Unit tests for date-of-birth parsing and age calculation."""

from __future__ import annotations

from datetime import date

import pytest

from people_analytics_service.person_age import (
    calculate_age,
    effective_age,
    parse_date_of_birth,
)


class TestParseDateOfBirth:
    def test_dmy_slash(self):
        assert parse_date_of_birth("26/9/2003") == date(2003, 9, 26)

    def test_dmy_dash(self):
        assert parse_date_of_birth("26-09-2003") == date(2003, 9, 26)

    def test_iso(self):
        assert parse_date_of_birth("2003-09-26") == date(2003, 9, 26)

    def test_invalid(self):
        assert parse_date_of_birth("not-a-date") is None


class TestCalculateAge:
    def test_before_birthday_this_year(self):
        dob = date(2003, 9, 26)
        assert calculate_age(dob, on=date(2026, 6, 20)) == 22

    def test_on_birthday(self):
        dob = date(2003, 9, 26)
        assert calculate_age(dob, on=date(2026, 9, 26)) == 23

    def test_effective_age_prefers_dob(self):
        assert effective_age(
            date_of_birth=date(2003, 9, 26),
            stored_age=99,
            on=date(2026, 6, 20),
        ) == 22

    def test_effective_age_legacy_fallback(self):
        assert effective_age(date_of_birth=None, stored_age=80) == 80
