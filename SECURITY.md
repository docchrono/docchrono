# Security policy

## Supported versions

Until the first stable release, security fixes target the latest published alpha version only.

## Reporting a vulnerability

Please use GitHub's private security-advisory workflow for the DocChrono repository. Do not post
vulnerability details in a public issue. Include affected versions, a minimal reproduction, impact,
and any suggested mitigation. The maintainers will acknowledge a complete report as soon as
practical and coordinate disclosure after a fix is available.

DocChrono parses untrusted files. A report involving parser resource exhaustion, path traversal,
embedded-content execution, provenance forgery, or unsafe deserialization is in scope.

Default builds enforce bounded traversal, input size, extracted text, DOCX expansion, PDF word
boxes, and entity-resolution candidate counts. These limits reduce denial-of-service risk but are
not a substitute for OS-level CPU/memory isolation when processing hostile files at scale.
