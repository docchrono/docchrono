import pytest

from docchrono.domain import Entity, EntityType, Event, EventType, Relationship
from docchrono.errors import AmbiguousReferenceError
from docchrono.views import EvidenceGraph


def _entity(identifier: str, name: str, *aliases: str) -> Entity:
    return Entity(
        id=identifier,
        type=EntityType.PERSON,
        canonical_name=name,
        aliases=aliases,
        mention_ids=(f"mention-{identifier}",),
        score=0.9,
    )


def _event(identifier: str, title: str) -> Event:
    return Event(
        id=identifier,
        type=EventType.ACTION,
        title=title,
        claim_ids=(f"claim-{identifier}",),
        score=0.8,
    )


def test_graph_returns_typed_deterministic_paths_with_claim_provenance() -> None:
    alice = _entity("entity-a", "Alice", "A. Smith")
    bob = _entity("entity-b", "Bob")
    approval = _event("event-approval", "Approval")
    approved = Relationship(
        id="relationship-approved",
        source_id=alice.id,
        target_id=approval.id,
        type="APPROVED",
        supporting_claim_ids=("claim-support",),
        opposing_claim_ids=("claim-opposes",),
        score=0.9,
    )
    notified = Relationship(
        id="relationship-notified",
        source_id=approval.id,
        target_id=bob.id,
        type="NOTIFIED",
        supporting_claim_ids=("claim-notified",),
        score=0.8,
    )
    graph = EvidenceGraph((bob, alice), (approval,), (notified, approved))

    assert graph.neighbors("A. Smith") == (approval,)
    assert graph.relationships("entity-a", "Approval") == (approved,)
    path = graph.find_path("Alice", "Bob")
    assert path == (approved, notified)
    assert path is not None
    assert path[0].supporting_claim_ids == ("claim-support",)
    assert path[0].opposing_claim_ids == ("claim-opposes",)


def test_exact_name_lookup_rejects_ambiguity() -> None:
    first = _entity("entity-a", "Alex")
    second = _entity("entity-b", "Alex")
    graph = EvidenceGraph((second, first), (), ())

    with pytest.raises(AmbiguousReferenceError, match="entity-a, entity-b"):
        graph.neighbors("Alex")

    assert graph.neighbors("entity-a") == ()
