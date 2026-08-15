from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from typing import TYPE_CHECKING, Any, cast

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

if TYPE_CHECKING:
    from spacy.tokens import Doc


@dataclass(frozen=True, slots=True)
class NlpIdentity:
    name: str
    version: str
    fallback: bool


class EnglishNlpPipeline:
    """Deterministic, offline-only English NLP pipeline.

    A caller may pass an already loaded Language object or the import name of a
    packaged spaCy model. Failure to find that package falls back locally unless
    ``require_model`` is true. This class never calls spaCy's download helpers.
    """

    def __init__(
        self,
        model: str | Language | None = None,
        *,
        require_model: bool = False,
        max_length: int = 2_000_000,
    ) -> None:
        self._fallback = False
        if isinstance(model, Language):
            self._nlp = model
        elif model:
            try:
                self._nlp = spacy.load(model)
            except (ImportError, OSError):
                if require_model:
                    raise
                self._nlp = self._blank_pipeline()
                self._fallback = True
        else:
            self._nlp = self._blank_pipeline()
            self._fallback = True
        self._nlp.max_length = max_length

    @staticmethod
    def _blank_pipeline() -> Language:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?", "\n"]})
        ruler = cast(
            EntityRuler,
            nlp.add_pipe(
                "entity_ruler",
                config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"},
            ),
        )
        # The fallback deliberately favors bounded, explainable forms. Broad
        # title-case matching is handled as a review-scored rule downstream.
        ruler.add_patterns(
            [
                {
                    "label": "PERSON",
                    "id": "honorific-person",
                    "pattern": [
                        {"LOWER": {"IN": ["mr.", "mrs.", "ms.", "dr.", "prof."]}},
                        {"IS_TITLE": True, "OP": "+"},
                    ],
                },
                {
                    "label": "ORG",
                    "id": "organization-suffix",
                    "pattern": [
                        {"IS_TITLE": True, "OP": "+"},
                        {
                            "LOWER": {
                                "IN": [
                                    "inc",
                                    "inc.",
                                    "llc",
                                    "ltd",
                                    "ltd.",
                                    "corp",
                                    "corp.",
                                    "corporation",
                                    "company",
                                    "bank",
                                    "university",
                                ]
                            }
                        },
                    ],
                },
            ]
        )
        return nlp

    @property
    def identity(self) -> NlpIdentity:
        meta: dict[str, Any] = self._nlp.meta
        name = str(meta.get("name") or "blank-en")
        version = str(meta.get("version") or distribution_version("spacy"))
        return NlpIdentity(name=name, version=version, fallback=self._fallback)

    @property
    def fallback(self) -> bool:
        return self._fallback

    def process(self, text: str) -> Doc:
        return self._nlp(text)
