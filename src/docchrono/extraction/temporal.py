from __future__ import annotations

import calendar
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from typing import Any, cast

import dateparser

from docchrono.domain import Document, TemporalExpression, TemporalPrecision
from docchrono.extraction.contracts import TemporalCandidate

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"

# Candidate detection is intentionally finite. dateparser never scans arbitrary
# document text; it only normalizes spans accepted by one of these patterns.
_TEMPORAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "range",
        re.compile(
            rf"\b(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})"
            rf"\s+(?:to|through|until|\u2013|\u2014)\s+"
            rf"(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instant",
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(?::\d{2})?"
            r"(?:\s*(?:Z|[+-]\d{2}:?\d{2}|[A-Z]{2,5}))?\b"
        ),
    ),
    (
        "instant",
        re.compile(
            rf"\b(?:{_WEEKDAY},?\s+)?\d{{1,2}}\s+{_MONTH}\s+\d{{4}}\s+"
            r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:Z|[+-]\d{4}|[A-Z]{2,5}))?\b",
            re.IGNORECASE,
        ),
    ),
    ("day", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    (
        "day",
        re.compile(
            rf"\b(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|"
            rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{4}})\b",
            re.IGNORECASE,
        ),
    ),
    ("month", re.compile(rf"\b{_MONTH}\s+\d{{4}}\b", re.IGNORECASE)),
    ("relative", re.compile(r"\b(?:today|yesterday|tomorrow)\b", re.IGNORECASE)),
    (
        "relative",
        re.compile(
            rf"\b(?:last|next|this)\s+(?:{_WEEKDAY}|week|month|year)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "relative",
        re.compile(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
            r"(?:day|days|week|weeks|month|months|year|years)\s+"
            r"(?:ago|later|before|after)\b",
            re.IGNORECASE,
        ),
    ),
    ("year", re.compile(r"(?<![\w-])(?:19|20)\d{2}(?![\w-])")),
)

_REFERENCE_KEYS = (
    "reference_datetime",
    "document_datetime",
    "date",
    "sent_at",
    "created_at",
)
_DETERMINISTIC_PARSE_BASE = datetime(2000, 1, 1)


class TemporalNormalizer:
    name = "docchrono.temporal"
    version = "1"

    def find(
        self, document: Document, *, default_timezone: str | None = None
    ) -> tuple[TemporalCandidate, ...]:
        candidates: list[tuple[int, int, str]] = []
        for kind, pattern in _TEMPORAL_PATTERNS:
            for match in pattern.finditer(document.normalized_text):
                candidates.append((match.start(), match.end(), kind))

        # Prefer the longest and most specific candidate at an overlapping span.
        selected: list[tuple[int, int, str]] = []
        for start, end, kind in sorted(
            candidates, key=lambda item: (item[0], -(item[1] - item[0]))
        ):
            if any(
                start < chosen_end and end > chosen_start
                for chosen_start, chosen_end, _ in selected
            ):
                continue
            selected.append((start, end, kind))

        reference = self._reference_datetime(document.metadata, default_timezone)
        results: list[TemporalCandidate] = []
        for start, end, kind in sorted(selected):
            original = document.normalized_text[start:end]
            expression = self._normalize(
                original,
                kind=kind,
                reference=reference,
                default_timezone=default_timezone,
            )
            score = 0.98 if expression.resolved and kind != "year" else 0.92
            if not expression.resolved:
                score = 0.65
            results.append(
                TemporalCandidate(
                    normalized_start=start,
                    normalized_end=end,
                    expression=expression,
                    score=score,
                )
            )
        return tuple(results)

    def _normalize(
        self,
        original: str,
        *,
        kind: str,
        reference: datetime | None,
        default_timezone: str | None,
    ) -> TemporalExpression:
        is_relative = kind == "relative"
        if is_relative and reference is None:
            return TemporalExpression(
                original_text=original,
                precision=TemporalPrecision.UNRESOLVED,
                timezone=default_timezone,
                is_relative=True,
                resolved=False,
            )

        if kind == "range":
            parts = re.split(
                r"\s+(?:to|through|until|\u2013|\u2014)\s+",
                original,
                maxsplit=1,
                flags=re.I,
            )
            if len(parts) == 2:
                left = self._parse(parts[0], reference, default_timezone)
                right = self._parse(parts[1], reference, default_timezone)
                if left is not None and right is not None:
                    return TemporalExpression(
                        original_text=original,
                        start=self._iso(left, TemporalPrecision.DAY),
                        end=self._iso(right, TemporalPrecision.DAY),
                        precision=TemporalPrecision.RANGE,
                        timezone=self._timezone_name(left, default_timezone),
                        is_relative=False,
                        resolved=True,
                    )

        parsed = self._parse(
            original,
            reference,
            default_timezone,
            strict=kind not in {"month", "year"},
        )
        if parsed is None:
            return TemporalExpression(
                original_text=original,
                precision=TemporalPrecision.UNRESOLVED,
                timezone=default_timezone,
                is_relative=is_relative,
                resolved=False,
            )

        precision = {
            "instant": TemporalPrecision.INSTANT,
            "day": TemporalPrecision.DAY,
            "month": TemporalPrecision.MONTH,
            "year": TemporalPrecision.YEAR,
            "relative": TemporalPrecision.DAY,
        }.get(kind, TemporalPrecision.UNRESOLVED)
        start, end = self._bounds(parsed, precision)
        return TemporalExpression(
            original_text=original,
            start=start,
            end=end,
            precision=precision,
            timezone=self._timezone_name(parsed, default_timezone),
            is_relative=is_relative,
            resolved=True,
        )

    @staticmethod
    def _parse(
        text: str,
        reference: datetime | None,
        default_timezone: str | None,
        *,
        strict: bool = True,
    ) -> datetime | None:
        settings: dict[str, Any] = {
            "DATE_ORDER": "MDY",
            "PREFER_DAY_OF_MONTH": "first",
            "PREFER_DATES_FROM": "past",
            "STRICT_PARSING": strict,
        }
        if reference is not None:
            settings["RELATIVE_BASE"] = reference
        if default_timezone:
            settings["TIMEZONE"] = default_timezone
            settings["RETURN_AS_TIMEZONE_AWARE"] = True
        return dateparser.parse(text, languages=["en"], settings=settings)

    @classmethod
    def _reference_datetime(
        cls,
        metadata: Mapping[str, object],
        default_timezone: str | None,
    ) -> datetime | None:
        values: list[object] = []
        for key in _REFERENCE_KEYS:
            value = metadata.get(key)
            if value is not None:
                values.append(value)
        email_obj = metadata.get("email")
        if isinstance(email_obj, Mapping):
            email = cast(Mapping[object, object], email_obj)
            for key in ("date", "sent_at"):
                value = email.get(key)
                if value is not None:
                    values.append(value)
        headers_obj = metadata.get("headers")
        if isinstance(headers_obj, Mapping):
            headers = cast(Mapping[object, object], headers_obj)
            header_dates = headers.get("date")
            if isinstance(header_dates, (tuple, list)):
                date_values = cast(tuple[object, ...] | list[object], header_dates)
                values.extend(date_values)
            elif header_dates is not None:
                values.append(header_dates)
        for value in values:
            if isinstance(value, datetime):
                return value
            if isinstance(value, date):
                return datetime.combine(value, time.min)
            if isinstance(value, str):
                settings: dict[str, Any] = {
                    "RELATIVE_BASE": _DETERMINISTIC_PARSE_BASE,
                    "RETURN_AS_TIMEZONE_AWARE": bool(default_timezone),
                    "STRICT_PARSING": True,
                }
                if default_timezone:
                    settings["TIMEZONE"] = default_timezone
                parsed = dateparser.parse(value, languages=["en"], settings=settings)
                if parsed is not None:
                    return parsed
        return None

    @staticmethod
    def _bounds(value: datetime, precision: TemporalPrecision) -> tuple[str, str]:
        if precision == TemporalPrecision.YEAR:
            return date(value.year, 1, 1).isoformat(), date(value.year, 12, 31).isoformat()
        if precision == TemporalPrecision.MONTH:
            last = calendar.monthrange(value.year, value.month)[1]
            return date(value.year, value.month, 1).isoformat(), date(
                value.year, value.month, last
            ).isoformat()
        rendered = TemporalNormalizer._iso(value, precision)
        return rendered, rendered

    @staticmethod
    def _iso(value: datetime, precision: TemporalPrecision) -> str:
        if precision == TemporalPrecision.INSTANT:
            return value.isoformat()
        return value.date().isoformat()

    @staticmethod
    def _timezone_name(value: datetime, default_timezone: str | None) -> str | None:
        if value.tzinfo is None:
            return default_timezone
        name = value.tzname()
        if name in {"UTC", "GMT"} or value.utcoffset() == UTC.utcoffset(value):
            return "UTC"
        return name or default_timezone
