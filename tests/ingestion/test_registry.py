from pathlib import Path

import pytest

from docchrono.domain import CaseConfig
from docchrono.errors import AdapterConflictError, AdapterContractError
from docchrono.ingestion import LoadedDocument, LoaderRegistry, LoadRequest


class ExampleLoader:
    name = "example"
    version = "1.0"
    supported_suffixes = (".example",)
    media_type = "text/example"

    def load(self, request: LoadRequest) -> LoadedDocument:
        return LoadedDocument(raw_text="ok", media_type=self.media_type)


def test_registry_supports_deterministic_clone_and_lookup() -> None:
    registry = LoaderRegistry((ExampleLoader(),))
    clone = LoaderRegistry(registry.loaders)

    assert clone.supported_suffixes == (".example",)
    assert clone.loader_for(".EXAMPLE") is not None
    assert clone.adapter_versions == {"example": "1.0"}


def test_registry_rejects_conflicting_suffix() -> None:
    class ConflictLoader(ExampleLoader):
        name = "conflict"

    registry = LoaderRegistry((ExampleLoader(),))

    with pytest.raises(AdapterConflictError, match="already handled"):
        registry.register(ConflictLoader())


def test_registry_can_explicitly_replace_conflicting_adapter() -> None:
    class ReplacementLoader(ExampleLoader):
        name = "replacement"

    registry = LoaderRegistry((ExampleLoader(),))

    registry.register(ReplacementLoader(), replace=True)

    assert [loader.name for loader in registry.loaders] == ["replacement"]
    selected = registry.loader_for(".example")
    assert selected is not None
    assert selected.name == "replacement"


def test_registry_rejects_noncanonical_suffix() -> None:
    class BadLoader(ExampleLoader):
        supported_suffixes = ("EXAMPLE",)

    with pytest.raises(AdapterContractError, match="suffixes"):
        LoaderRegistry((BadLoader(),))


def test_example_loader_fixture_is_well_formed() -> None:
    request = LoadRequest(
        path=Path("source.example"),
        content=b"ok",
        content_sha256="0" * 64,
        suffix=".example",
        config=CaseConfig(),
    )

    assert ExampleLoader().load(request).raw_text == "ok"
