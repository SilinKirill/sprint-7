"""Build a local FAISS index from task-2/knowledge_base only."""
import argparse
import hashlib
import importlib.metadata
import json
import time

from common import BASE, ROOT, KB, OUTPUT, MODEL, DIMENSION, QUERY_PROMPT, chunk_document, load_model, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--offline', action='store_true', help='Use cached model files only')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if OUTPUT.exists():
        raise SystemExit('task-3/index already exists; preserve or move it before rebuilding.')
    files = sorted(KB.glob('*.md'))
    if len(files) != 30:
        raise SystemExit('Expected 30 Markdown files in task-2/knowledge_base.')
    started = time.perf_counter()
    chunks, sources = [], []
    for path in files:
        # No dictionary, prepared source text, original wiki metadata or cache is loaded.
        if path.is_symlink():
            raise ValueError('Symlink sources are not supported')
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
        source = path.relative_to(ROOT).as_posix()
        chunks.extend(chunk_document(text, source))
        sources.append({'source': source, 'sha256': hashlib.sha256(raw).hexdigest()})
    print(f'Prepared {len(files)} documents, {len(chunks)} chunks.', flush=True)
    load_started = time.perf_counter()
    model = load_model(args.device, offline=args.offline)
    model_load_seconds = time.perf_counter() - load_started
    revision = getattr(model[0].auto_model.config, '_commit_hash', None)
    if not revision:
        raise ValueError('Could not determine model revision for reproducible query encoding')
    embedding_started = time.perf_counter()
    vectors = encode(model, [c['title'] + '\n\n' + c['text'] for c in chunks],
                     batch_size=args.batch_size)
    embedding_seconds = time.perf_counter() - embedding_started
    import faiss
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(vectors)
    if index.ntotal != len(chunks):
        raise ValueError('Index/metadata count mismatch')
    OUTPUT.mkdir()
    # Serialize FAISS to bytes: works with Unicode Windows paths as well.
    (OUTPUT / 'faiss.index').write_bytes(faiss.serialize_index(index).tobytes())
    with (OUTPUT / 'chunks.jsonl').open('w', encoding='utf-8') as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    report = {'model': MODEL, 'model_revision': revision, 'dimension': DIMENSION,
              'device': args.device, 'batch_size': args.batch_size,
              'index_type': 'IndexFlatIP', 'metric': 'cosine (normalized vectors)',
              'query_prompt': QUERY_PROMPT, 'document_prompt': '',
              'document_input': 'title + two newlines + chunk text',
              'documents': len(files), 'chunks': len(chunks),
              'splitter': 'RecursiveCharacterTextSplitter',
              'chunk_size_words': 300, 'minimum_chunk_words': 100,
              'small_chunk_handling': 'extend with adjacent source context; may overlap',
              'min_chunk_words': min(c['word_count'] for c in chunks),
              'max_chunk_words': max(c['word_count'] for c in chunks),
              'model_load_seconds': round(model_load_seconds, 3),
              'embedding_seconds': round(embedding_seconds, 3),
              'total_seconds': round(time.perf_counter() - started, 3),
              'sources': sources,
              'packages': {name: importlib.metadata.version(name) for name in
                           ('sentence-transformers', 'transformers', 'torch', 'numpy', 'faiss-cpu', 'langchain-text-splitters')}}
    report['artifact_sha256'] = {name: hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()
                                 for name in ('faiss.index', 'chunks.jsonl')}
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved task-3/index: {index.ntotal} vectors x {DIMENSION}.')
    print(f'Embeddings: {embedding_seconds:.1f} s; total: {report["total_seconds"]:.1f} s.')


if __name__ == '__main__':
    main()
