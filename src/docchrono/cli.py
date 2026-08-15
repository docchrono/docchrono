from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from docchrono import __version__
from docchrono.case import Case
from docchrono.domain import Event
from docchrono.errors import DocChronoError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docchrono",
        description="Build source-linked chronologies and evidence graphs from documents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="analyze documents and optionally save the case")
    build.add_argument("sources", nargs="+", type=Path, help="files or directories to analyze")
    build.add_argument("-o", "--output", type=Path, help="write the case as deterministic JSON")
    build.add_argument(
        "--strict",
        action="store_true",
        help="stop when the first document cannot be processed",
    )
    build.add_argument("--json", action="store_true", help="print the summary as JSON")

    inspect = commands.add_parser("inspect", help="show a saved case summary")
    inspect.add_argument("case", type=Path)
    inspect.add_argument("--json", action="store_true", help="print the summary as JSON")

    timeline = commands.add_parser("timeline", help="print the timeline from a saved case")
    timeline.add_argument("case", type=Path)
    timeline.add_argument("--json", action="store_true", help="print timeline records as JSON")

    return parser


def _summary(case: Case) -> dict[str, object]:
    failures = len(case.report.failures) + sum(
        len(result.failures) for result in case.report.documents
    )
    return {
        "complete": case.report.complete,
        "documents": len(case.documents),
        "source_references": len(case.source_references),
        "entities": len(case.entities),
        "claims": len(case.claims),
        "events": len(case.events),
        "relationships": len(case.relationships),
        "review_items": len(case.review_items),
        "document_failures": failures,
    }


def _print_summary(case: Case, *, as_json: bool) -> None:
    summary = _summary(case)
    if as_json:
        print(json.dumps(summary, sort_keys=True))
        return
    print(f"Documents processed: {summary['documents']}")
    print(f"People and organizations discovered: {summary['entities']}")
    print(f"Events discovered: {summary['events']}")
    print(f"Relationships discovered: {summary['relationships']}")
    print(f"Items needing review: {summary['review_items']}")
    if summary["document_failures"]:
        print(f"Document failures: {summary['document_failures']}")
    print(f"Chronology generated: {'yes' if case.report.complete else 'partial'}")
    print(f"Evidence graph generated: {'yes' if case.report.complete else 'partial'}")


def _event_sources(case: Case, event: Event) -> tuple[str, ...]:
    document_by_id = {document.id: document for document in case.documents}
    source_by_id = {source.id: source for source in case.source_references}
    names: set[str] = set()
    for span in case.evidence(event):
        document = document_by_id.get(span.document_id)
        if document is None:
            continue
        names.update(
            source_by_id[source_id].filename
            for source_id in document.source_reference_ids
            if source_id in source_by_id
        )
    return tuple(sorted(names, key=lambda value: (value.casefold(), value)))


def _timeline_record(case: Case, event: Event) -> dict[str, object]:
    record = event.model_dump(mode="json")
    assert isinstance(record, dict)
    record["sources"] = list(_event_sources(case, event))
    return record


def _print_timeline(case: Case, *, as_json: bool) -> None:
    events = case.timeline.all
    if as_json:
        print(
            json.dumps(
                [_timeline_record(case, event) for event in events],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return

    for event in events:
        dates = [temporal.start for temporal in event.temporal if temporal.resolved]
        print(min(date for date in dates if date is not None) if any(dates) else "Undated")
        print(event.title)
        sources = _event_sources(case, event)
        print(f"Sources: {', '.join(sources) if sources else '(unavailable)'}")
        print()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            case = Case.build(args.sources, strict=args.strict)
            if args.output is not None:
                saved = case.save(args.output)
                print(f"Saved: {saved}")
            _print_summary(case, as_json=args.json)
            return 0 if case.report.complete else 2
        if args.command == "inspect":
            _print_summary(Case.load(args.case), as_json=args.json)
            return 0
        if args.command == "timeline":
            _print_timeline(Case.load(args.case), as_json=args.json)
            return 0
    except DocChronoError as exc:
        print(f"docchrono: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
