# Keep the default core offline and deterministic

The default package performs no runtime downloads, telemetry, or cloud calls and records the rules, model, configuration, and dependency versions that produced a Case. Optional integrations may extend the package later, but generative AI, semantic embeddings, and remote graph stores are outside the v0.1 core so identical inputs can produce reproducible evidence-linked output.
