"""Advanced orchestration and adapter APIs.

Most callers should use :meth:`docchrono.Case.build`.  This namespace exposes
the pipeline only for callers that need custom document loaders or extractors.
"""

from docchrono.advanced.pipeline import Pipeline

__all__ = ["Pipeline"]
