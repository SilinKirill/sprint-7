"""Incremental vectors, full consistent FAISS snapshot, and append-only update log."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np
from shared6 import BASE, ROOT, MODEL, DIMENSION, QUERY_PROMPT, chunk_document, load_base, load_model, encode
from settings6 import load_settings
from snapshot import digest, read_snapshot, write_snapshot


@contextmanager
def update_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)


def collect_documents(source):
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError('Source directory missing; run init_source.py first')
    documents = {}
    for p in sorted(source.rglob('*')):
        if p.is_symlink() or (p.exists() and not p.resolve().is_relative_to(source)):
            raise ValueError('Source symlinks or junctions are not supported')
        if p.is_file() and p.suffix.lower() in ('.md', '.txt'):
            documents[p.relative_to(source).as_posix()] = p.read_bytes()
    return documents


def input_key(chunk):
    return digest((chunk['title'] + '\n\n' + chunk['text']).encode('utf-8'))


def synchronize(source, output, source_prefix, base, embed):
    """Pure workflow with injectable encoder/base, tested without any private files."""
    source, output = Path(source), Path(output)
    documents = collect_documents(source)
    fingerprints = {name: digest(raw) for name, raw in documents.items()}
    exists = output.exists()
    old_index, old_chunks, old_report = read_snapshot(output) if exists else base()
    previous = old_report.get('files', {}) if exists else {}
    changes = {'files_added': len(fingerprints.keys() - previous.keys()),
               'files_changed': sum(fingerprints[k] != previous[k] for k in fingerprints.keys() & previous.keys()),
               'files_removed': len(previous.keys() - fingerprints.keys())}
    if (exists and fingerprints == previous and old_report.get('source_prefix') == source_prefix and
            old_report.get('chunking_version') == 'task3-100-300-v1'):
        return {**changes, 'status': 'unchanged', 'new_chunks': 0, 'embeddings_generated': 0, 'reused_vectors': len(old_chunks),
                'index_size': len(old_chunks)}
    if old_report['model'] != MODEL or old_report['dimension'] != DIMENSION or old_report['query_prompt'] != QUERY_PROMPT:
        raise ValueError('Base encoder mismatch')
    revision = old_report['model_revision']
    chunks = []
    for name, raw in documents.items():
        source_id = source_prefix.rstrip('/') + '/' + name
        items = chunk_document(raw.decode('utf-8-sig'), source_id)
        for c in items:
            c['chunk_id'] = digest(source_id.encode())[:16] + ':' + str(c['chunk_index'])
        chunks.extend(items)
    old_vectors = old_index.reconstruct_n(0, old_index.ntotal)
    old_by_id = {c['chunk_id']: input_key(c) for c in old_chunks}
    cache = {input_key(c): old_vectors[i] for i, c in enumerate(old_chunks)}
    missing = {input_key(c): c for c in chunks if input_key(c) not in cache}
    if missing:
        values = list(missing.values())
        vectors = embed([c['title'] + '\n\n' + c['text'] for c in values], revision)
        if (vectors.shape != (len(values), DIMENSION) or not np.isfinite(vectors).all() or
                not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
            raise ValueError('Invalid new embeddings')
        cache.update(zip(missing, vectors))
    matrix = np.asarray([cache[input_key(c)] for c in chunks], dtype='float32').reshape((-1, DIMENSION))
    index = faiss.IndexFlatIP(DIMENSION)
    if chunks:
        index.add(matrix)
    metadata = {'model': MODEL, 'dimension': DIMENSION, 'query_prompt': QUERY_PROMPT,
                'model_revision': revision, 'files': fingerprints, 'source_prefix': source_prefix,
                'chunking_version': 'task3-100-300-v1', 'documents': len(documents)}
    # Do not publish a snapshot while its source has changed during embedding.
    if {k: digest(v) for k, v in collect_documents(source).items()} != fingerprints:
        raise ValueError('Source changed during update; rerun to capture a consistent version')
    write_snapshot(output, index, chunks, metadata)
    return {**changes, 'status': 'updated',
            'new_chunks': sum(old_by_id.get(c['chunk_id']) != input_key(c) for c in chunks),
            'embeddings_generated': len(missing),
            'reused_vectors': len(chunks) - sum(input_key(c) in missing for c in chunks),
            'index_size': len(chunks)}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    started = time.perf_counter()
    entry = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'error',
             'files_added': 0, 'files_changed': 0, 'files_removed': 0, 'new_chunks': 0,
             'index_size': None, 'errors': []}
    settings = None
    output = BASE / 'index/snapshot.zip'
    model = None

    def embed(texts, revision):
        nonlocal model
        if model is None:
            model = load_model(settings.embedding_device, revision=revision, offline=settings.offline_embeddings)
        return encode(model, texts)

    failed = False
    try:
        settings = load_settings()
        with update_lock(BASE / 'index/.update.lock'):
            entry.update(synchronize(BASE / 'incoming', output, 'task-6/incoming', load_base, embed))
    except Exception as exc:
        failed = True
        entry['errors'] = [type(exc).__name__ + ': ' + str(exc)]
    finally:
        entry['finished_at'] = datetime.now(timezone.utc).isoformat()
        entry['seconds'] = round(time.perf_counter() - started, 3)
        directory = BASE / 'logs'
        directory.mkdir(exist_ok=True)
        with (directory / 'updates.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(json.dumps(entry, ensure_ascii=True))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
