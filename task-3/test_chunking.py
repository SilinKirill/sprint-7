"""Synthetic tests only. Never reads the user's knowledge base."""
import unittest
import re

from common import chunk_document


class ChunkingTests(unittest.TestCase):
    def check_coverage(self, text, chunks):
        body_start = text.index('\n')
        for word in re.finditer(r'\S+', text[body_start:]):
            start, end = word.start() + body_start, word.end() + body_start
            self.assertTrue(any(c['start_char'] <= start and end <= c['end_char'] for c in chunks))
        self.assertEqual(len({c['chunk_id'] for c in chunks}), len(chunks))
        for c in chunks:
            self.assertEqual(c['text'], text[c['start_char']:c['end_char']])
            self.assertEqual(c['start_line'], text.count('\n', 0, c['start_char']) + 1)
            self.assertEqual(c['end_line'], text.count('\n', 0, c['end_char']) + 1)

    def test_sentence_boundaries_and_offsets(self):
        sentence = ' '.join(['synthetic'] * 19) + ' end.'
        text = '# Example title\n\n' + '\n\n'.join([sentence] * 20)
        chunks = chunk_document(text, 'fake/doc.md')
        self.check_coverage(text, chunks)
        self.assertTrue(all(100 <= c['word_count'] <= 300 for c in chunks))
        self.assertTrue(all(c['text'].endswith('end.') for c in chunks))

    def test_short_tail_gets_adjacent_context(self):
        text = '# Example\n\n' + ' '.join(['word'] * 315)
        chunks = chunk_document(text, 'fake/doc.md')
        self.check_coverage(text, chunks)
        self.assertEqual(len(chunks), 2)
        self.assertLess(chunks[1]['start_char'], chunks[0]['end_char'])
        self.assertTrue(all(100 <= c['word_count'] <= 300 for c in chunks))

    def test_long_sentence_fallback(self):
        text = '# Example\n\n' + ' '.join(['word'] * 710)
        chunks = chunk_document(text, 'fake/doc.md')
        self.check_coverage(text, chunks)
        self.assertTrue(all(100 <= c['word_count'] <= 300 for c in chunks))

    def test_unicode_crlf_and_determinism(self):
        text = '# Synthetic Ω\r\n\r\n' + ' '.join(['café'] * 180)
        chunks = chunk_document(text, 'fake/doc.md')
        self.check_coverage(text, chunks)
        self.assertEqual(chunks, chunk_document(text, 'fake/doc.md'))
        self.assertEqual(chunks[0]['title'], 'Synthetic Ω')

    def test_empty_document_rejected(self):
        with self.assertRaises(ValueError):
            chunk_document('# Empty\n\n', 'fake/doc.md')

    def test_small_section_and_repeated_sentences(self):
        text = '# Example\n\n## Section\n\n' + ('Repeated sentence has five words. ' * 95)
        chunks = chunk_document(text, 'fake/doc.md')
        self.check_coverage(text, chunks)
        self.assertTrue(all(100 <= c['word_count'] <= 300 for c in chunks))

    def test_unequal_paragraphs(self):
        text = '# Example\n\n' + '\n\n'.join(' '.join([f'word{i}'] * n) + '.'
                                              for i, n in enumerate([45, 290, 35, 180]))
        chunks = chunk_document(text, 'fake/doc.md')
        self.check_coverage(text, chunks)
        self.assertTrue(all(100 <= c['word_count'] <= 300 for c in chunks))


if __name__ == '__main__':
    unittest.main()
