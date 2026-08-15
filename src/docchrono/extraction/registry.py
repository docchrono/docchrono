from __future__ import annotations

import re
from collections.abc import Iterable
from typing import cast

from docchrono.errors import AdapterConflictError, AdapterContractError
from docchrono.extraction.contracts import Extractor

_ADAPTER_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ExtractorRegistry:
    """A name-unique registry with stable execution order."""

    def __init__(self, extractors: Iterable[Extractor] = ()) -> None:
        self._extractors: dict[str, Extractor] = {}
        for extractor in extractors:
            self.register(extractor)

    def register(self, extractor: Extractor, *, replace: bool = False) -> None:
        self._validate(extractor)
        key = extractor.name.casefold()
        if key in self._extractors and not replace:
            raise AdapterConflictError(f"extractor name already registered: {extractor.name}")
        self._extractors[key] = extractor

    def remove(self, name: str) -> None:
        self._extractors.pop(name.casefold(), None)

    @property
    def extractors(self) -> tuple[Extractor, ...]:
        return tuple(
            sorted(
                self._extractors.values(),
                key=lambda item: (item.order, item.name.casefold(), item.version),
            )
        )

    @staticmethod
    def _validate(extractor: object) -> None:
        if not isinstance(extractor, Extractor):
            raise AdapterContractError("extractor does not satisfy the Extractor protocol")
        name = cast(object, extractor.name)
        version = cast(object, extractor.version)
        if not isinstance(name, str) or not _ADAPTER_TOKEN.fullmatch(name):
            raise AdapterContractError("extractor name must be a stable non-empty token")
        if not isinstance(version, str) or not _ADAPTER_TOKEN.fullmatch(version):
            raise AdapterContractError("extractor version must be a stable non-empty token")
        order = cast(object, extractor.order)
        if not isinstance(order, int):
            raise AdapterContractError("extractor order must be an integer")
