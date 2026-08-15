from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from docchrono.domain import ReviewAction, ReviewDecision
from docchrono.domain.ids import stable_id
from docchrono.errors import ReviewDecisionError


def create_decision(
    action: ReviewAction,
    target_ids: Iterable[str],
    *,
    payload: Mapping[str, Any] | None = None,
    reason: str | None = None,
    supersedes: str | None = None,
) -> ReviewDecision:
    targets = tuple(sorted(set(target_ids)))
    if not targets:
        raise ReviewDecisionError("a review decision requires at least one target")
    content = dict(payload or {})
    identifier = decision_content_id(
        action,
        targets,
        payload=content,
        reason=reason,
        supersedes=supersedes,
    )
    return ReviewDecision(
        id=identifier,
        action=action,
        target_ids=targets,
        payload=content,
        reason=reason,
        supersedes=supersedes,
    )


def decision_content_id(
    action: ReviewAction,
    target_ids: Iterable[str],
    *,
    payload: Mapping[str, Any] | None = None,
    reason: str | None = None,
    supersedes: str | None = None,
) -> str:
    return stable_id(
        "review",
        action,
        tuple(sorted(set(target_ids))),
        dict(payload or {}),
        reason,
        supersedes,
    )


def expected_decision_id(decision: ReviewDecision) -> str:
    return decision_content_id(
        decision.action,
        decision.target_ids,
        payload=decision.payload,
        reason=decision.reason,
        supersedes=decision.supersedes,
    )
