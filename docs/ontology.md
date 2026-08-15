# DocChrono ontology

DocChrono version 0.1 uses a small, code-level evidence ontology. It is designed to answer:

> Who did what, when, with whom, according to which exact source passage?

## Core terms

| Term | Meaning |
|---|---|
| `SourceReference` | One observed source path and its content hash. |
| `Document` | Parsed immutable content; identical bytes may have multiple source references. |
| `EvidenceSpan` | Exact raw-text interval with optional page, paragraph, field, and boxes. |
| `Mention` | A source expression that may refer to an entity. |
| `Claim` | A source-scoped assertion with polarity, modality, participants, time, and evidence. |
| `Entity` | A conservatively resolved identity backed by mentions. |
| `Event` | A provisional grouping of compatible event claims. |
| `Relationship` | A derived view over supporting and opposing claims. |
| `ReviewItem` | An ambiguity or low-confidence item requiring a human decision. |
| `ReviewDecision` | An immutable, replayable human action. |

## Epistemic rules

1. A claim describes what a source says; it does not assert objective truth.
2. Conflicting claims remain parallel records.
3. Every accepted finding must resolve to source evidence.
4. A relationship cannot exist without at least one supporting claim.
5. Entity identity does not depend on the mutable display name.
6. Unresolved time is retained rather than guessed.
7. Ambiguous entity or event grouping becomes review work instead of a silent merge.

## Vocabulary

Version 0.1 includes people, organizations, locations, documents, and other entities; and
communication, meeting, decision, transaction, action, and other events. Relationship predicates
are normalized strings derived from source claims, allowing controlled extensions without a
breaking schema change.

This is not a formal OWL ontology. RDF/JSON-LD mappings, SPARQL, custom ontology registries, and
reasoner integration are possible future adapters, not version 0.1 commitments.
