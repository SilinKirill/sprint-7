"""Manually verify retrieval or an answer against the updated snapshot."""
import argparse
from settings6 import load_settings
from runtime6 import LiveRAG, SnapshotRetriever
from shared6 import BASE, rag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True)
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    settings = load_settings()
    if args.sources_only:
        retriever = SnapshotRetriever(BASE / 'index/snapshot.zip', settings)
        for p in retriever.find(args.query):
            print(p['passage_id'], p['source'], round(p['cosine_score'], 4))
        return
    result = LiveRAG(settings).ask(args.query)
    print(result.get('reply', result.get('error', 'No response')))
    if result['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
