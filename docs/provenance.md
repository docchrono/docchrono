# Provenance contract

Source-linked evidence is DocChrono's primary integrity boundary.

## Required guarantees

- Original file bytes are SHA-256 hashed before parsing.
- Raw extracted text is immutable in a case snapshot.
- Normalized NLP text carries a monotonic boundary map back to raw text.
- An evidence span's quote equals `document.raw_text[raw_start:raw_end]`.
- Normalized evidence offsets map to the same raw interval.
- Page, paragraph, sentence, field, and bounding boxes are retained when a loader can supply them.
- Mentions reference evidence spans; claims reference evidence spans and participants; entities,
  events, and relationships trace through those records.
- Full save/load verifies referential and quotation integrity before returning a `Case`.

Normalization never replaces the source text. It produces a separate analysis string and an
offset map. A finding is rejected if an adapter emits an invalid or non-round-trippable span.

The saved payload SHA-256 detects accidental corruption; it is not a signature and does not
authenticate a deliberately forged case. Source re-verification and signed case envelopes are
outside version 0.1. `save_sanitized()` is an explicit redaction export: it retains evidence
quotes but removes document text, so quotation round-tripping cannot be re-verified from that
artifact and it must not be treated as a full provenance-preserving case.

## Typed failures

Unsupported, missing, encrypted, malformed, oversized, and image-only sources are never silently
skipped. The build report retains a typed failure such as `UNSUPPORTED_FORMAT`, `ENCRYPTED`,
`MALFORMED`, `LIMIT_EXCEEDED`, or `OCR_REQUIRED`.

Non-strict builds can return useful findings from successfully processed documents while marking
the report incomplete. Strict builds raise `BuildFailed` with the partial report attached.
