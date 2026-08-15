from __future__ import annotations

from docchrono.domain import OffsetMap


def normalize_text(raw_text: str) -> tuple[str, OffsetMap]:
    """Normalize line endings and retain an exact boundary map to raw text.

    The map contains one raw boundary for every normalized boundary. A CRLF is
    represented by one LF whose boundaries span both raw code points. No other
    lossy whitespace or Unicode transformation is performed.
    """

    normalized: list[str] = []
    transitions: list[tuple[int, int]] = [(0, 0)]
    raw_index = 0
    normalized_index = 0
    previous_delta = 0
    while raw_index < len(raw_text):
        character = raw_text[raw_index]
        if character == "\r":
            raw_index += 1
            if raw_index < len(raw_text) and raw_text[raw_index] == "\n":
                raw_index += 1
            normalized.append("\n")
            normalized_index += 1
            delta = raw_index - normalized_index
            if delta != previous_delta:
                transitions.append((normalized_index, delta))
                previous_delta = delta
            continue
        normalized.append(character)
        raw_index += 1
        normalized_index += 1
    return "".join(normalized), OffsetMap(
        normalized_length=normalized_index,
        transitions=tuple(transitions),
    )


__all__ = ["normalize_text"]
