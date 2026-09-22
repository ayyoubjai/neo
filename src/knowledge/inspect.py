from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge.graph import KnowledgeGraph
from knowledge.ingest import write_report


def main():
    parser = argparse.ArgumentParser(description="Inspect source claims, proposed connections, and assessment history")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--claim", help="Explain one claim back to exact passages and investigation evidence")
    group.add_argument("--collection", help="List claims prioritized for investigation")
    parser.add_argument("--output", help="Also write a JSON artifact")
    args = parser.parse_args()
    graph = KnowledgeGraph(ensure_schema=False)
    try:
        result = graph.explain(args.claim) if args.claim else {"collection": args.collection, "claims": graph.queue(args.collection)}
        if args.output:
            write_report(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        graph.close()


if __name__ == "__main__":
    main()
