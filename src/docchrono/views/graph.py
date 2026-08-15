from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from itertools import pairwise
from typing import TypeAlias, cast

import networkx as nx

from docchrono.domain import Entity, Event, Relationship
from docchrono.errors import AmbiguousReferenceError

GraphNode: TypeAlias = Entity | Event
NodeReference: TypeAlias = GraphNode | str


class EvidenceGraph:
    """A typed, deterministic view over an internal NetworkX evidence graph."""

    __slots__ = ("_graph", "_names", "_nodes", "_relationships", "_undirected")

    _graph: nx.MultiDiGraph[str, dict[str, object], dict[str, object]]
    _undirected: nx.Graph[str, dict[str, object], dict[str, object]]

    def __init__(
        self,
        entities: Iterable[Entity],
        events: Iterable[Event],
        relationships: Iterable[Relationship],
    ) -> None:
        nodes = tuple(sorted((*entities, *events), key=lambda node: node.id))
        node_map: dict[str, GraphNode] = {}
        for node in nodes:
            if node.id in node_map:
                raise ValueError(f"duplicate graph node id: {node.id}")
            node_map[node.id] = node

        ordered_relationships = tuple(sorted(relationships, key=_relationship_sort_key))
        relationship_ids: set[str] = set()
        graph = cast(
            "nx.MultiDiGraph[str, dict[str, object], dict[str, object]]",
            nx.MultiDiGraph(),
        )
        graph.add_nodes_from(node_map.keys())
        undirected = cast(
            "nx.Graph[str, dict[str, object], dict[str, object]]",
            nx.Graph(),
        )
        undirected.add_nodes_from(node_map.keys())
        for relationship in ordered_relationships:
            if relationship.id in relationship_ids:
                raise ValueError(f"duplicate relationship id: {relationship.id}")
            relationship_ids.add(relationship.id)
            if relationship.source_id not in node_map:
                raise ValueError(
                    f"relationship {relationship.id} references unknown source "
                    f"{relationship.source_id}"
                )
            if relationship.target_id not in node_map:
                raise ValueError(
                    f"relationship {relationship.id} references unknown target "
                    f"{relationship.target_id}"
                )
            graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                key=relationship.id,
                relationship=relationship,
            )
            undirected.add_edge(relationship.source_id, relationship.target_id)

        names: dict[str, set[str]] = {}
        for node in nodes:
            for name in _node_names(node):
                names.setdefault(name, set()).add(node.id)

        self._nodes = node_map
        self._relationships = ordered_relationships
        self._graph = graph
        self._undirected = undirected
        self._names = {name: tuple(sorted(ids)) for name, ids in names.items()}

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        return tuple(self._nodes[node_id] for node_id in sorted(self._nodes))

    def neighbors(
        self,
        node: NodeReference,
        *,
        relationship: str | None = None,
    ) -> tuple[GraphNode, ...]:
        node_id = self._resolve(node)
        neighbor_ids: set[str] = set()
        for item in self._incident_relationships(node_id):
            if relationship is not None and item.type != relationship:
                continue
            other_id = item.target_id if item.source_id == node_id else item.source_id
            neighbor_ids.add(other_id)
        return tuple(self._nodes[item] for item in sorted(neighbor_ids))

    def relationships(
        self,
        left: NodeReference,
        right: NodeReference | None = None,
    ) -> tuple[Relationship, ...]:
        left_id = self._resolve(left)
        if right is None:
            return self._incident_relationships(left_id)

        right_id = self._resolve(right)
        return tuple(
            item
            for item in self._incident_relationships(left_id)
            if {item.source_id, item.target_id} == {left_id, right_id}
            or (left_id == right_id == item.source_id == item.target_id)
        )

    def find_path(
        self,
        start: NodeReference,
        end: NodeReference,
    ) -> tuple[Relationship, ...] | None:
        start_id = self._resolve(start)
        end_id = self._resolve(end)
        if start_id == end_id:
            return ()

        parents: dict[str, str | None] = {start_id: None}
        queue: deque[str] = deque([start_id])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(self._undirected.neighbors(current)):
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                if neighbor == end_id:
                    queue.clear()
                    break
                queue.append(neighbor)

        if end_id not in parents:
            return None

        node_path: list[str] = []
        cursor: str | None = end_id
        while cursor is not None:
            node_path.append(cursor)
            cursor = parents[cursor]
        node_path.reverse()

        path: list[Relationship] = []
        for left_id, right_id in pairwise(node_path):
            candidates = self.relationships(left_id, right_id)
            path.append(min(candidates, key=_relationship_sort_key))
        return tuple(path)

    def _resolve(self, reference: NodeReference) -> str:
        if isinstance(reference, (Entity, Event)):
            if reference.id not in self._nodes:
                raise KeyError(f"graph does not contain node id {reference.id!r}")
            return reference.id
        if reference in self._nodes:
            return reference
        candidates = self._names.get(reference, ())
        if not candidates:
            raise KeyError(f"no graph node has id or exact name {reference!r}")
        if len(candidates) > 1:
            joined = ", ".join(candidates)
            raise AmbiguousReferenceError(
                f"exact name {reference!r} refers to multiple graph nodes: {joined}"
            )
        return next(iter(candidates))

    def _incident_relationships(self, node_id: str) -> tuple[Relationship, ...]:
        return tuple(
            item
            for item in self._relationships
            if item.source_id == node_id or item.target_id == node_id
        )


def _node_names(node: GraphNode) -> tuple[str, ...]:
    if isinstance(node, Entity):
        return tuple(dict.fromkeys((node.canonical_name, *node.aliases)))
    return (node.title,)


def _relationship_sort_key(relationship: Relationship) -> tuple[str, ...]:
    return (
        relationship.source_id,
        relationship.target_id,
        relationship.type,
        relationship.id,
    )
