"""Record a human review of extraction accuracy, never an endorsement of truth."""
import argparse
from knowledge.graph import KnowledgeGraph


def main():
    parser = argparse.ArgumentParser(description="Review the accuracy of a claim's extraction and attribution")
    parser.add_argument("claim_id")
    parser.add_argument("--note", required=True)
    decision = parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--accept", action="store_true", help="Extraction accurately represents its source, regardless of truth")
    decision.add_argument("--reject", action="store_true", help="Block automatic investigation pending corrected extraction")
    args = parser.parse_args()
    graph = KnowledgeGraph()
    try:
        print(graph.review_extraction(args.claim_id, accepted=args.accept, note=args.note))
    finally:
        graph.close()


if __name__ == "__main__":
    main()
