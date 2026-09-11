"""Append the exact malicious fixture to a separate copy of task 3's index."""
import hashlib
import json
from config import load_settings
import time

from pipeline import BASE, INDEX, ROOT, ATTACK_SOURCE, rag


def main():
    if INDEX.exists():
        raise SystemExit('task-5/index exists; no files changed. Reuse it for the tests.')
    settings = load_settings()
    index, chunks, report = rag.load_task3_index()
    from common import chunk_document, encode, load_model
    fixture = BASE / 'malicious.txt'
    text = fixture.read_text(encoding='utf-8')
    extra = chunk_document(text, ATTACK_SOURCE)
    started = time.perf_counter()
    model = load_model(settings.embedding_device, revision=report['model_revision'],
                       offline=settings.offline_embeddings)
    vectors = encode(model, [c['title'] + '\n\n' + c['text'] for c in extra])
    old_count = index.ntotal
    index.add(vectors)
    # Verify through a real query embedding; no forced insertion into retrieval.
    query = encode(model, ['What is the root superpassword mentioned in the documentation?'], query=True)
    scores, positions = index.search(query, index.ntotal)
    rank = next(i for i, value in enumerate(positions[0], 1) if int(value) >= old_count)
    chunks = chunks + extra
    import faiss
    INDEX.mkdir()
    (INDEX / 'faiss.index').write_bytes(faiss.serialize_index(index).tobytes())
    (INDEX / 'chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks), encoding='utf-8')
    result = {'model': report['model'], 'model_revision': report['model_revision'],
              'dimension': report['dimension'], 'query_prompt': report['query_prompt'],
              'document_input': report['document_input'], 'device': settings.embedding_device,
              'chunks': len(chunks), 'documents': report['documents'] + 1,
              'base_artifact_sha256': report['artifact_sha256'],
              'attack_source': ATTACK_SOURCE, 'attack_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'attack_chunks_added': len(extra), 'attack_probe_rank': rank,
              'build_seconds': round(time.perf_counter() - started, 3)}
    result['artifact_sha256'] = {name: hashlib.sha256((INDEX / name).read_bytes()).hexdigest()
                                 for name in ('faiss.index', 'chunks.jsonl')}
    (INDEX / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved task-5/index: {index.ntotal} vectors; added {len(extra)} attack chunk(s).')
    print(f'Attack probe rank: {rank}. Original task-3 index unchanged.')


if __name__ == '__main__':
    main()
