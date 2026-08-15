# DocChrono

DocChrono turns a bounded collection of documents into an evidence-linked account of entities, claims, events, relationships, and chronology without treating extracted statements as established facts.

## Source material

**Case**:
A bounded collection of source material, derived findings, and review decisions considered together.
_Avoid_: Workspace, project, corpus

**Source Reference**:
A location at which original document bytes were encountered. Multiple source references may point to the same document.
_Avoid_: Document path, source document

**Document**:
Unique source content identified independently of its filename or location.
_Avoid_: File

**Evidence Span**:
An exact, recoverable region of a document that supports an extracted interpretation.
_Avoid_: Citation, snippet

## Interpretation

**Mention**:
An evidence span interpreted as a surface-form reference to an entity.
_Avoid_: Entity occurrence

**Claim**:
A source-scoped assertion that preserves what was stated, including polarity, modality, time, and evidence, without declaring it true.
_Avoid_: Fact, finding

**Entity**:
A canonical referent provisionally associated with one or more mentions.
_Avoid_: Name, actor

**Event**:
A provisional occurrence represented by one or more compatible event claims.
_Avoid_: Claim, timeline item

**Relationship**:
A derived view of compatible claims connecting entities or events; it is never an independently asserted fact.
_Avoid_: Edge, fact

## Analysis

**Timeline**:
An ordered view of dated events plus an explicit undated collection.
_Avoid_: Chronology record

**Evidence Graph**:
A navigable view of entities and events whose connections retain their supporting and opposing claims and evidence.
_Avoid_: Knowledge graph, truth graph

**Review Item**:
An uncertain automated interpretation offered for human judgment.
_Avoid_: Error, task

**Review Decision**:
An auditable and reversible human judgment that takes precedence over automated interpretation.
_Avoid_: Override, correction

**Extraction Score**:
An extractor-specific ranking signal that is not a probability unless separately calibrated.
_Avoid_: Confidence probability
