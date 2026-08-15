from __future__ import annotations

import re
from collections.abc import Iterable
from typing import cast

from docchrono.errors import AdapterConflictError, AdapterContractError

from .contracts import DocumentLoader

_ADAPTER_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_ADAPTER_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class LoaderRegistry:
    """Validated suffix-to-loader registry with deterministic lookup."""

    def __init__(self, loaders: Iterable[DocumentLoader] = ()) -> None:
        self._by_suffix: dict[str, DocumentLoader] = {}
        self._by_name: dict[str, DocumentLoader] = {}
        for loader in loaders:
            self.register(loader)

    def register(self, loader: DocumentLoader, *, replace: bool = False) -> None:
        _validate_loader(loader)
        suffixes = tuple(loader.supported_suffixes)
        conflicts: dict[str, DocumentLoader] = {}
        named_existing = self._by_name.get(loader.name)
        if named_existing is not None:
            conflicts[named_existing.name] = named_existing
        for suffix in suffixes:
            existing = self._by_suffix.get(suffix)
            if existing is not None:
                conflicts[existing.name] = existing

        if conflicts and not replace:
            if named_existing is not None:
                raise AdapterConflictError(f"loader name is already registered: {loader.name}")
            suffix = next(suffix for suffix in suffixes if suffix in self._by_suffix)
            existing = self._by_suffix[suffix]
            raise AdapterConflictError(f"suffix {suffix!r} is already handled by {existing.name!r}")

        for existing in conflicts.values():
            self._by_name.pop(existing.name, None)
            for existing_suffix, registered in tuple(self._by_suffix.items()):
                if registered is existing:
                    del self._by_suffix[existing_suffix]

        self._by_name[loader.name] = loader
        for suffix in suffixes:
            self._by_suffix[suffix] = loader

    def loader_for(self, suffix: str) -> DocumentLoader | None:
        return self._by_suffix.get(suffix.lower())

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_suffix))

    @property
    def loaders(self) -> tuple[DocumentLoader, ...]:
        return tuple(self._by_name[name] for name in sorted(self._by_name))

    @property
    def adapter_versions(self) -> dict[str, str]:
        return {loader.name: loader.version for loader in self.loaders}

    @classmethod
    def with_builtins(cls) -> LoaderRegistry:
        # Local import keeps the public contract usable by small custom adapters
        # without importing every parser dependency.
        from .loaders import DocxLoader, EmlLoader, MarkdownLoader, PdfLoader, TextLoader

        return cls((TextLoader(), MarkdownLoader(), PdfLoader(), DocxLoader(), EmlLoader()))


def _validate_loader(loader: object) -> None:
    if not isinstance(loader, DocumentLoader):
        raise AdapterContractError("loader does not implement the DocumentLoader protocol")

    name = cast(object, loader.name)
    version = cast(object, loader.version)
    media_type = cast(object, loader.media_type)
    suffixes = cast(object, loader.supported_suffixes)
    if not isinstance(name, str) or not _ADAPTER_NAME.fullmatch(name):
        raise AdapterContractError(
            "loader.name must be a lowercase identifier containing letters, digits, '_' or '-'"
        )
    if not isinstance(version, str) or not _ADAPTER_VERSION.fullmatch(version):
        raise AdapterContractError("loader.version must be a non-empty stable version identifier")
    if not isinstance(media_type, str) or "/" not in media_type or media_type.strip() != media_type:
        raise AdapterContractError("loader.media_type must be a valid non-empty media type")
    if not isinstance(suffixes, tuple) or not suffixes:
        raise AdapterContractError("loader.supported_suffixes must be a non-empty tuple")
    suffix_values = cast(tuple[object, ...], suffixes)
    for suffix in suffix_values:
        if (
            not isinstance(suffix, str)
            or not suffix.startswith(".")
            or suffix != suffix.lower()
            or suffix.strip() != suffix
            or len(suffix) < 2
        ):
            raise AdapterContractError(
                "loader suffixes must be lowercase, trimmed, and begin with '.'"
            )
    typed_suffixes = cast(tuple[str, ...], suffix_values)
    if len(set(typed_suffixes)) != len(typed_suffixes):
        raise AdapterContractError("loader.supported_suffixes contains duplicates")


__all__ = ["LoaderRegistry"]
