"""One atomic ZIP snapshot: FAISS bytes, chunk metadata and their manifest."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import faiss
import numpy as np
from shared6 import MODEL, DIMENSION, QUERY_PROMPT


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_snapshot(path, index, chunks, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index_bytes = faiss.serialize_index(index).tobytes()
    text_bytes = (''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks)).encode('utf-8')
    report = {**metadata, 'chunks': len(chunks),
              'written_at': datetime.now(timezone.utc).isoformat(),
              'artifact_sha256': {'faiss.index': digest(index_bytes), 'chunks.jsonl': digest(text_bytes)}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('faiss.index', index_bytes)
        z.writestr('chunks.jsonl', text_bytes)
        z.writestr('report.json', json.dumps(report, ensure_ascii=False))
    fd, temporary = tempfile.mkstemp(prefix='snapshot-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        # Validate the complete replacement before publishing it.
        read_snapshot(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report


def read_snapshot(path):
    data = Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        report = json.loads(z.read('report.json'))
        payloads = {name: z.read(name) for name in ('faiss.index', 'chunks.jsonl')}
    for name, content in payloads.items():
        if digest(content) != report['artifact_sha256'][name]:
            raise ValueError('Snapshot checksum mismatch')
    if (report['model'] != MODEL or report['dimension'] != DIMENSION or
            report['query_prompt'] != QUERY_PROMPT or not report['model_revision']):
        raise ValueError('Snapshot encoder configuration mismatch')
    chunks = [json.loads(line) for line in payloads['chunks.jsonl'].decode('utf-8').splitlines()]
    index = faiss.deserialize_index(np.frombuffer(payloads['faiss.index'], dtype='uint8').copy())
    if (index.ntotal != len(chunks) or len(chunks) != report['chunks'] or index.d != DIMENSION or
            index.metric_type != faiss.METRIC_INNER_PRODUCT):
        raise ValueError('Snapshot index and chunks disagree')
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Duplicate chunk IDs')
    return index, chunks, {**report, 'snapshot_sha256': digest(data)}
