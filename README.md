# DocChrono

DocChrono turns English documents into an evidence-linked chronology and graph without
requiring a cloud AI service or API key.

```python
from docchrono import Case

case = Case.build("./documents")

print(case.timeline)
print(case.entities)
print(case.graph.neighbors("Robert Williams"))
```

DocChrono treats extracted statements as **source claims, not verified facts**. Every claim,
event, entity, and relationship can be traced back to exact text in an input document.

> [!IMPORTANT]
> DocChrono is pre-release alpha software. Review extracted findings before relying on them.

## What version 0.1 does

- Reads `.txt`, `.md`, text-based `.pdf`, `.docx`, and RFC 822 `.eml` files.
- Uses a finite, rule-based English spaCy pipeline to extract candidate people, organizations,
  locations, dates, claims, and provisional events.
- Resolves conservative duplicate-entity candidates without an external LLM.
- Builds a chronology and an evidence graph.
- Retains raw offsets, page/field context where available, and exact source quotations.
- Records low-confidence candidates and typed document failures for review.
- Produces deterministic identifiers and stable saved JSON in the same environment.

DocChrono includes both chronology and a lightweight ontology: its typed domain model defines
documents, evidence, mentions, claims, entities, events, and relationships. Version 0.1 does not
implement OWL, RDF, SPARQL, or a general-purpose ontology editor.

## Installation

DocChrono supports CPython 3.11, 3.12, and 3.13.

```bash
python -m pip install docchrono
```

For development from a checkout:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"  # Windows
```

On macOS or Linux, use `.venv/bin/python` instead.

## Python API

Build and save a case:

```python
from docchrono import Case, CaseConfig

case = Case.build(
    ["./emails", "./reports"],
    config=CaseConfig(
        auto_accept_threshold=0.90,
        review_threshold=0.60,
        max_extracted_chars=1_000_000,
    ),
)

case.save("investigation.docchrono.json")
```

Load without parsing the sources again:

```python
case = Case.load("investigation.docchrono.json")

for event in case.timeline:
    print(event.title, event.temporal)
```

Inspect provenance:

```python
claim = case.claims[0]

for span in case.evidence(claim):
    print(span.quote, span.page, span.field)
```

Query the evidence graph:

```python
neighbors = case.graph.neighbors("Robert Williams")
path = case.graph.find_path("Robert Williams", "Payment #932")
```

Review decisions are immutable and auditable; applying decisions returns a new case:

```python
pending = case.review.pending
reviewed = case.review.apply(
    [case.review.accept(pending[0].id, reason="Confirmed against the source")]
)
```

## Command line

```bash
docchrono build ./documents --output case.json
docchrono inspect case.json
docchrono timeline case.json
```

Add `--strict` to fail the build if any document cannot be processed, or `--json` for
machine-readable output.

## Offline and deterministic by default

The standard pipeline makes no network calls. It combines finite parsing/extraction rules, a
blank English spaCy tokenizer, dateparser, RapidFuzz, and NetworkX. It does not bundle a
pretrained statistical language model, so version 0.1 favors inspectability and precision over
broad recall. Identical bytes, configuration, rules, model, and dependency versions produce
identical semantic identifiers and ordering.

PyMuPDF is not installed or selected by the standard pipeline because it is AGPL-3.0 or
commercially licensed. A future opt-in adapter must keep that choice and its obligations explicit.

## Deliberate version 0.1 limits

- English only.
- No OCR; a PDF containing a page with images but no extractable text is reported as
  `OCR_REQUIRED` rather than partially analyzed.
- No spreadsheets, Outlook `.msg`, chat UI, semantic search, contradiction engine, or graph
  visualization yet.
- Relationships are derived views over source claims, never independently asserted facts.
- Event grouping and entity resolution are conservative and can create review items.
- Per-file, extracted-text, archive, traversal, and entity-resolution budgets reject oversized
  work with typed limit failures; advanced callers can tune those budgets with `CaseConfig`.

See [the architecture](docs/architecture.md), [ontology](docs/ontology.md), and
[provenance contract](docs/provenance.md) for the design details.

## Contributing and security

Development instructions are in [CONTRIBUTING.md](CONTRIBUTING.md). Please report security
issues using [SECURITY.md](SECURITY.md), not a public issue.

DocChrono is licensed under the [Apache License 2.0](LICENSE).
