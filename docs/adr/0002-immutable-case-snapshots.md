# Expose immutable complete Case snapshots

The ordinary interface constructs a complete immutable Case with `Case.build(...)` or restores one with `Case.load(...)`; partially built state belongs to the advanced Pipeline. This departs from a mutable `Case(...).build()` interface to prevent stale derived views, make review application reversible, and keep stage ordering and invalidation behind one deep module seam.
