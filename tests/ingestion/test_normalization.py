from docchrono.ingestion import normalize_text


def test_normalization_retains_lossless_raw_boundaries() -> None:
    raw = "alpha\r\nbeta\rgamma\n"

    normalized, boundaries = normalize_text(raw)

    assert normalized == "alpha\nbeta\ngamma\n"
    assert len(boundaries) == len(normalized) + 1
    first_newline = normalized.index("\n")
    assert raw[boundaries[first_newline] : boundaries[first_newline + 1]] == "\r\n"
    second_newline = normalized.index("\n", first_newline + 1)
    assert raw[boundaries[second_newline] : boundaries[second_newline + 1]] == "\r"
    assert boundaries == tuple(sorted(boundaries))
    assert boundaries[-1] == len(raw)


def test_normalization_is_identity_for_lf_text() -> None:
    normalized, boundaries = normalize_text("a\nb")

    assert normalized == "a\nb"
    assert boundaries == (0, 1, 2, 3)
