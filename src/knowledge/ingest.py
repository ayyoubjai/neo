from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from autonomy.cognition_client import CognitionClient
from knowledge.documents import read_document
from knowledge.extraction import compare_claims, extract_passage
from knowledge.graph import KnowledgeGraph
from knowledge.matching import rank_candidates


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def ingest(paths: list[Path], *, collection: str, client, graph=None, report_path: Path, ocr: bool = False) -> dict:
    report = {"collection": collection, "persisted": False, "documents": [], "issues": []}
    # Also serves report-only imports, without requiring a graph service.
    batch_claims = {}
    aliases = await asyncio.to_thread(graph.aliases) if graph is not None and hasattr(graph, "aliases") else []
    for path in paths:
        entry = {"source": str(path), "claims": [], "relationships": [], "skipped_passages": 0, "issues": []}
        report["documents"].append(entry)
        previous_claims = dict(batch_claims)
        try:
            document = await asyncio.to_thread(read_document, path, ocr=ocr)
            entry.update({"document_id": document.id, "sha256": document.sha256})
            results = []
            for passage_index, passage in enumerate(document.passages):
                if graph is not None and await asyncio.to_thread(graph.passage_complete, passage.id):
                    entry["skipped_passages"] += 1
                    continue
                try:
                    claims, issues = await extract_passage(client, passage, context=document.passages[max(0, passage_index-1):passage_index] + document.passages[passage_index+1:passage_index+2])
                except Exception as exc:
                    claims, issues = [], [{"passage_id": passage.id, "reason": str(exc)}]
                for claim in claims:
                    candidates = list(batch_claims.values())
                    if graph is not None:
                        candidates += await asyncio.to_thread(graph.candidates, claim, collection)
                    candidates = rank_candidates(claim, candidates, aliases)
                    try:
                        links, link_issues = await compare_claims(client, claim, candidates)
                        entry["relationships"].extend(links)
                        issues.extend(link_issues)
                    except Exception as exc:
                        issues.append({"claim_id": claim["id"], "reason": f"Comparison failed: {exc}"})
                    batch_claims[claim["id"]] = claim
                entry["claims"].extend(claims)
                entry["issues"].extend(issues)
                results.append({"passage": passage, "claims": claims, "issues": issues})
            if graph is not None:
                await asyncio.to_thread(graph.save_document, document, collection, results, entry["relationships"])
                entry["persisted"] = True
                report["persisted"] = True
            entry["status"] = "review" if entry["issues"] else "complete"
        except Exception as exc:
            batch_claims = previous_claims
            entry["status"] = "failed"
            entry["issues"].append({"reason": str(exc)})
        report["issues"].extend(entry["issues"])
        write_report(report_path, report)
    return report


async def run(args):
    paths = []
    for raw in args.paths:
        path = Path(raw)
        paths.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".markdown", ".pdf", ".epub"}) if path.is_dir() else [path])
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if not paths:
        raise ValueError("No supported documents found")
    client = CognitionClient()
    await client.wait_until_available()
    graph = None if args.report_only else KnowledgeGraph()
    try:
        report = await ingest(paths, collection=args.collection, client=client, graph=graph, report_path=Path(args.report), ocr=args.ocr)
    finally:
        if graph is not None:
            graph.close()
    review_count = sum(c["review_required"] for d in report["documents"] for c in d["claims"])
    print(f"Processed {len(report['documents'])} documents; {len(report['issues'])} issues; {review_count} claims need extraction review. Report: {args.report}")
    return 1 if report["issues"] else 0


def main():
    parser = argparse.ArgumentParser(description="Import source-attributed claims without adopting them as beliefs")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--report", default="data/knowledge/ingestion-report.json")
    parser.add_argument("--ocr", action="store_true", help="Locally OCR PDF pages without text; requires pdftoppm and tesseract")
    parser.add_argument("--report-only", action="store_true", help="Extract a review report without writing to Neo4j")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
