"""Search the saved local index; output passages, not an LLM-generated answer."""
import argparse
import hashlib
import json

from common import BASE, OUTPUT, MODEL, DIMENSION, QUERY_PROMPT, load_model, encode

EXAMPLES = [
    'Which unit was afraid of flying and compensated by excelling at ground maintenance?',
    'Which medical assistant could use twenty retractable manipulator arms simultaneously?',
    'Which unit harassed a trooper by locking doors and running over their feet?',
]


def load_index():
    import faiss
    import numpy as np
    report = json.loads((OUTPUT / 'report.json').read_text(encoding='utf-8'))
    if report['model'] != MODEL or report['dimension'] != DIMENSION or report['query_prompt'] != QUERY_PROMPT:
        raise ValueError('Index embedding configuration differs from this script')
    for name in ('faiss.index', 'chunks.jsonl'):
        if hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() != report['artifact_sha256'][name]:
            raise ValueError('Index artifact changed: ' + name)
    index = faiss.deserialize_index(np.frombuffer((OUTPUT/'faiss.index').read_bytes(), dtype='uint8').copy())
    chunks = [json.loads(line) for line in (OUTPUT/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    if index.ntotal != len(chunks) or len(chunks) != report['chunks'] or index.d != DIMENSION:
        raise ValueError('Index/metadata mismatch')
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError('Unexpected index metric')
    return index, chunks, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--query')
    group.add_argument('--examples', action='store_true')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.top_k < 1 or (args.query is not None and not args.query.strip()):
        parser.error('Use a nonempty query and positive --top-k')
    index, chunks, report = load_index()
    model = load_model(args.device, revision=report['model_revision'], offline=args.offline)
    queries = EXAMPLES if args.examples else [args.query]
    vectors = encode(model, queries, query=True)
    scores, positions = index.search(vectors, min(args.top_k, index.ntotal))
    results = []
    for query, row_scores, row_ids in zip(queries, scores, positions):
        hits = [{'rank': rank, 'cosine_score': float(score), **chunks[int(idx)]}
                for rank, (score, idx) in enumerate(zip(row_scores, row_ids), 1)]
        results.append({'query': query, 'hits': hits})
        print('\nQUERY:', query)
        for hit in hits:
            print(f'\n[{hit["rank"]}] {hit["source"]}:{hit["start_line"]} '
                  f'chunk={hit["chunk_id"]} cosine={hit["cosine_score"]:.4f}')
            print(hit['title'])
            print(hit['text'])
    if args.examples:
        (BASE/'search_examples.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('\nSaved task-3/search_examples.json. Check passage relevance locally.')


if __name__ == '__main__':
    main()
