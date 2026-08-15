# Architecture

DocChrono is an offline, deterministic pipeline with one deep ordinary interface:

```text
source discovery
  -> parsing and normalized-to-raw offset maps
  -> mention, time, and claim extraction
  -> conservative entity resolution
  -> provisional event and relationship derivation
  -> chronology and evidence graph views
  -> integrity-checked persistence
```

`Case` is an immutable snapshot. Callers never observe a partially built instance. Review
decisions return a new snapshot and are retained as a deterministic log.

## Module boundaries

- `docchrono.domain` owns immutable Pydantic records and enumerations.
- `docchrono.ingestion` owns source discovery, hashing, loaders, deduplication, parsing, and raw
  provenance.
- `docchrono.extraction` owns spaCy/rule processing, temporal normalization, thresholds, and
  evidence-span construction.
- `docchrono.resolution` owns alias clustering, provisional events, and relationships derived
  from claims.
- `docchrono.views` owns read-only chronology and graph queries.
- `docchrono.review` owns decision construction and deterministic replay.
- `docchrono.persistence` owns schema-aware, atomic JSON save/load and integrity validation.
- `docchrono.advanced` orchestrates stages and exposes loader/extractor seams.

The standard pipeline has no remote dependencies. A user-supplied Python adapter is trusted
code and may, independently of DocChrono, perform arbitrary I/O.

## Determinism boundary

Semantic identifiers are hashes of canonical semantic inputs rather than timestamps or random
UUIDs. Source traversal, record collections, timeline output, graph queries, failures, and review
items all use stable ordering with an identifier tie-breaker.

The build manifest records DocChrono, Python, dependency, adapter, model, and configuration
versions. Reproducibility requires those inputs to match. Machine/runtime metadata is descriptive
and is not used to create semantic identifiers.

## Extension seams

Document loaders receive immutable bytes and return loader-neutral parsed text plus structural
segments. Extractors receive normalized documents and return offset-based candidates. The core
validates adapter names, versions, scores, offsets, referenced candidates, and provenance before
creating public domain records.

The public seams are intentionally narrow. Graph databases, LLMs, OCR systems, and alternative
storage engines are not part of the standard version 0.1 pipeline.
