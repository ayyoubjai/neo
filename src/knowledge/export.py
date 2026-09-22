"""Export a self-contained, offline explorer; source text is never executable HTML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge.graph import KnowledgeGraph


def render_explorer(records: list[dict], collection: str, truncated: bool = False) -> str:
    payload = json.dumps({"collection": collection, "records": records, "truncated": truncated}, ensure_ascii=True).replace("<", "\\u003c")
    template = Path(__file__).with_name("explorer.html").read_text(encoding="utf-8")
    template = template.replace("__GRAPH_SCRIPT__", Path(__file__).with_name("explorer-graph.js").read_text(encoding="utf-8"))
    return template.replace("__KNOWLEDGE_DATA__", payload)


def main():
    parser = argparse.ArgumentParser(description="Export source, concept, and assessment views to offline HTML")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", default="data/knowledge/explorer.html")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.limit <= 2000:
        parser.error("--limit must be between 1 and 2000")
    graph = KnowledgeGraph(ensure_schema=False)
    try:
        rows = graph.queue(args.collection, limit=args.limit + 1)
        records = [graph.explain(row["id"]) for row in rows[:args.limit]]
    finally:
        graph.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_explorer(records, args.collection, len(rows) > args.limit), encoding="utf-8")
    print(f"Exported {len(records)} claims to {output}")


if __name__ == "__main__":
    main()
