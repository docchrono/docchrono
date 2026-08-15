# Third-party licenses

DocChrono is distributed under Apache-2.0. Its dependencies remain under their own licenses.
This summary is provided for convenience; the installed distributions contain authoritative
license texts and metadata.

| Dependency | Purpose | License |
|---|---|---|
| Beautiful Soup | HTML text parsing | MIT |
| dateparser | Date and time normalization | BSD-3-Clause |
| NetworkX | In-memory graph construction and traversal | BSD-3-Clause |
| pdfplumber | PDF text and layout extraction | MIT |
| Pydantic | Domain-model validation | MIT |
| python-docx | DOCX parsing | MIT |
| RapidFuzz | Explainable fuzzy entity-resolution candidates | MIT |
| spaCy | English tokenization and rule-based NLP pipeline | MIT |

Transitive dependencies retain their own licenses. DocChrono does not bundle a third-party
pretrained language model or evaluation corpus in the runtime wheel.

PyMuPDF is intentionally not a DocChrono dependency. PyMuPDF is offered under AGPL-3.0 or a
commercial license; any future opt-in integration will carry a separate explicit notice.
