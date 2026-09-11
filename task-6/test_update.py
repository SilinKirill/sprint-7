"""Real FAISS and filesystem, synthetic texts and vectors; no model or private files."""
import hashlib
from pathlib import Path
import shutil
import uuid
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np
from shared6 import MODEL, DIMENSION, QUERY_PROMPT, chunk_document, rag
from snapshot import read_snapshot
from update_index import synchronize, update_lock
from runtime6 import SnapshotRetriever

TEXT = '# Synthetic unit\n\nThe synthetic unit inspects storage equipment and records maintenance dates. Its battery lasts for exactly twenty days before a mandatory replacement at the station.'


def vectors(texts, revision):
    result = []
    for text in texts:
        rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        vector = rng.normal(size=DIMENSION).astype('float32')
        result.append(vector / np.linalg.norm(vector))
    return np.asarray(result, dtype='float32').reshape((-1, DIMENSION))


def base():
    return faiss.IndexFlatIP(DIMENSION), [], {'model': MODEL, 'dimension': DIMENSION,
                                            'query_prompt': QUERY_PROMPT, 'model_revision': 'synthetic-revision'}


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.test_area = Path(__file__).resolve().parent / '.test-work'
        self.root = self.test_area / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.source = self.root / 'incoming'
        self.source.mkdir()
        self.output = self.root / 'index/snapshot.zip'
        self.embed = Mock(side_effect=vectors)

    def tearDown(self):
        if not self.root.resolve().is_relative_to(self.test_area.resolve()):
            raise ValueError('Unsafe test cleanup path')
        shutil.rmtree(self.root)

    def update(self):
        return synchronize(self.source, self.output, 'synthetic/incoming', base, self.embed)

    def test_add_no_change_edit_delete(self):
        p = self.source / 'one.md'
        p.write_text(TEXT, encoding='utf-8')
        first = self.update()
        self.assertEqual((first['files_added'], first['new_chunks']), (1, 1))
        saved = self.output.read_bytes()
        self.embed.reset_mock()
        self.assertEqual(self.update()['status'], 'unchanged')
        self.embed.assert_not_called()
        self.assertEqual(saved, self.output.read_bytes())
        p.write_text(TEXT.replace('twenty days', 'thirty days'), encoding='utf-8')
        self.assertEqual(self.update()['files_changed'], 1)
        self.assertIn('thirty days', read_snapshot(self.output)[1][0]['text'])
        p.unlink()
        self.embed.reset_mock()
        self.assertEqual(self.update()['files_removed'], 1)
        self.assertEqual(read_snapshot(self.output)[0].ntotal, 0)
        self.embed.assert_not_called()

    def test_initial_base_vectors_reused(self):
        (self.source / 'one.md').write_text(TEXT, encoding='utf-8')
        idx, _, report = base()
        chunks = chunk_document(TEXT, 'old/one.md')
        idx.add(vectors([c['title'] + '\n\n' + c['text'] for c in chunks], 'synthetic-revision'))
        result = synchronize(self.source, self.output, 'synthetic/incoming', lambda: (idx, chunks, report), self.embed)
        self.assertEqual(result['embeddings_generated'], 0)
        self.embed.assert_not_called()

    def test_only_new_content_embedded_and_unique_ids(self):
        (self.source / 'same.md').write_text(TEXT, encoding='utf-8')
        self.update()
        (self.source / 'same.txt').write_text(TEXT.replace('twenty', 'forty'), encoding='utf-8')
        self.embed.reset_mock()
        result = self.update()
        self.assertEqual((result['new_chunks'], result['reused_vectors']), (1, 1))
        chunks = read_snapshot(self.output)[1]
        self.assertEqual(len({c['chunk_id'] for c in chunks}), 2)

    def test_embedding_failure_preserves_old_snapshot(self):
        p = self.source / 'one.md'
        p.write_text(TEXT, encoding='utf-8')
        self.update()
        before = self.output.read_bytes()
        p.write_text(TEXT.replace('twenty', 'forty'), encoding='utf-8')
        self.embed.side_effect = RuntimeError('synthetic encoder failure')
        with self.assertRaises(RuntimeError):
            self.update()
        self.assertEqual(before, self.output.read_bytes())

    def test_invalid_empty_document_preserves_snapshot(self):
        (self.source / 'one.md').write_text(TEXT, encoding='utf-8')
        self.update()
        before = self.output.read_bytes()
        (self.source / 'empty.md').write_text('# Empty', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.update()
        self.assertEqual(before, self.output.read_bytes())

    def test_source_change_during_embedding_aborts_commit(self):
        p = self.source / 'one.md'
        p.write_text(TEXT, encoding='utf-8')
        def changing(texts, revision):
            p.write_text(TEXT + ' Changed.', encoding='utf-8')
            return vectors(texts, revision)
        self.embed.side_effect = changing
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.update()
        self.assertFalse(self.output.exists())

    def test_failed_replace_leaves_old_snapshot(self):
        p = self.source / 'one.md'
        p.write_text(TEXT, encoding='utf-8')
        self.update()
        before = self.output.read_bytes()
        p.write_text(TEXT + ' Updated.', encoding='utf-8')
        with patch('snapshot.os.replace', side_effect=PermissionError('synthetic lock')):
            with self.assertRaises(PermissionError):
                self.update()
        self.assertEqual(before, self.output.read_bytes())
        self.assertEqual(list(self.output.parent.glob('*.tmp')), [])

    def test_concurrent_update_lock(self):
        lock = self.root / 'lock'
        with update_lock(lock):
            with self.assertRaises(FileExistsError):
                with update_lock(lock):
                    self.fail('Second update acquired the lock')
            self.assertTrue(lock.exists())
        self.assertFalse(lock.exists())

    def test_retriever_reloads_without_loading_model(self):
        p = self.source / 'one.md'
        p.write_text(TEXT, encoding='utf-8')
        self.update()
        retriever = SnapshotRetriever(self.output, rag.Settings())
        self.assertFalse(retriever.reload())
        p.write_text(TEXT.replace('twenty', 'ninety'), encoding='utf-8')
        self.update()
        self.assertTrue(retriever.reload())
        self.assertIn('ninety', retriever.chunks[0]['text'])
        self.assertIsNone(retriever.model)


if __name__ == '__main__':
    unittest.main()
