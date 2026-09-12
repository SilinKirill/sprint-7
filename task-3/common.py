"""Shared chunking and local embedding configuration. No files read on import."""
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
KB = ROOT / 'task-2' / 'knowledge_base'
OUTPUT = BASE / 'index'
MODEL = 'Qwen/Qwen3-Embedding-0.6B'
DIMENSION = 1024
QUERY_PROMPT = ('Instruct: Given a question, retrieve relevant passages from the '
                'knowledge base that answer the question.\nQuery: ')


def chunk_document(text, source, minimum=100, maximum=300):
    """Use LangChain; extend undersized fragments with adjacent source context.

    Character offsets always refer to the original decoded Markdown.
    Small fragments may overlap their neighbors to reach the minimum size.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from bisect import bisect_left

    if not 0 < minimum <= maximum:
        raise ValueError('Invalid chunk bounds')
    title_match = re.match(r'#\s+([^\n]+)', text)
    title = title_match.group(1).strip() if title_match else Path(source).stem
    body_start = title_match.end() if title_match else 0
    body = text[body_start:]
    words = list(re.finditer(r'\S+', body))
    if not words:
        raise ValueError('Empty document: ' + source)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=maximum, chunk_overlap=0,
        length_function=lambda value: len(value.split()),
        separators=[r'\n\s*\n', r'\n', r'(?<=[.!?])\s+', r'\s+'],
        is_separator_regex=True, keep_separator='end', strip_whitespace=True,
    )
    starts = [w.start() for w in words]
    spans, cursor = [], 0
    for fragment in splitter.split_text(body):
        start = body.find(fragment, cursor)
        if start < 0:
            raise ValueError('Splitter fragment is not an exact source slice')
        end = start + len(fragment)
        cursor = end
        first = bisect_left(starts, start)
        last = bisect_left(starts, end)
        if last - first < minimum:
            # Add real neighboring text instead of padding or a tiny tail.
            first = max(0, last - minimum)
            last = min(len(words), max(last, first + minimum))
            start, end = words[first].start(), words[last-1].end()
        if not spans or spans[-1] != (start, end):
            spans.append((start, end))
    chunks = []
    for ordinal, (start, end) in enumerate(spans):
        start, end = body_start + start, body_start + end
        count = len(text[start:end].split())
        if count > maximum or (count < minimum and len(words) >= minimum):
            raise ValueError('Chunk word count outside configured bounds')
        chunks.append({'chunk_id': f'{Path(source).stem}:{ordinal:03d}',
                       'source': source, 'title': title, 'chunk_index': ordinal,
                       'start_char': start, 'end_char': end,
                       'start_line': text.count('\n', 0, start) + 1,
                       'end_line': text.count('\n', 0, end) + 1,
                       'word_count': count, 'text': text[start:end]})
    return chunks


def load_model(device='cpu', revision=None, offline=False):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device=device, revision=revision,
                                local_files_only=offline, trust_remote_code=False)
    # Small chunks; reject oversized token sequences rather than silently truncate.
    model.max_seq_length = 8192
    if model.get_sentence_embedding_dimension() != DIMENSION:
        raise ValueError('Unexpected embedding dimension')
    return model


def encode(model, texts, *, query=False, batch_size=4):
    import numpy as np
    prompt = QUERY_PROMPT if query else ''
    for text in texts:
        length = len(model.tokenizer.encode(prompt + text, add_special_tokens=True))
        if length > model.max_seq_length:
            raise ValueError('Input exceeds configured token limit; no truncation allowed')
    vectors = model.encode(texts, prompt=prompt, batch_size=batch_size,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=not query)
    vectors = np.ascontiguousarray(vectors, dtype='float32')
    if vectors.shape != (len(texts), DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding matrix')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Expected normalized embeddings')
    return vectors
