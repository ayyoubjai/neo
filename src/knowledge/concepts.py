"""Register a reviewed concept alias without merging source claims."""
import argparse
from knowledge.graph import KnowledgeGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alias', required=True)
    parser.add_argument('--canonical', required=True)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    graph = KnowledgeGraph()
    try:
        graph.add_alias(args.alias, args.canonical, args.note)
    finally:
        graph.close()


if __name__ == '__main__':
    main()
