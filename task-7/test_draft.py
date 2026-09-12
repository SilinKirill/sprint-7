"""Synthetic checks only; no Ollama or corpus access."""
import unittest
import json
from pathlib import Path
import shutil
import uuid
from types import SimpleNamespace
from unittest.mock import Mock
from draft_golden import merge_case, safe_chunks, checkpoint, resume, failure_reason, generate, quote_options


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {'source': 'synthetic.md', 'title': 'Example-42',
                      'text': 'Example-42 requires calibration every seventeen days.'}
        self.case = {'id': 'known_6', 'kind': 'known', 'question': '', 'topic': '',
                     'expected_sources': [], 'reference_quotes': [{'source': '', 'text': ''}]}
        self.generated = {'question': 'How often is calibration required?', 'topic': 'Calibration',
                          'expected_answer': 'Every seventeen days.', 'answer_keywords': ['seventeen days'],
                          'quote': self.chunk['text']}

    def test_valid_quote_and_source_assigned_without_mutating_input(self):
        result = merge_case(self.case, self.generated, 'synthetic.md', [self.chunk])
        self.assertEqual(result['reference_quotes'][0]['text'], self.chunk['text'])
        self.assertEqual(result['expected_sources'], ['synthetic.md'])
        self.assertEqual(self.case['question'], '')

    def test_invented_quote_rejected(self):
        self.generated['quote'] = 'Calibration happens every ninety days.'
        with self.assertRaises(ValueError):
            merge_case(self.case, self.generated, 'synthetic.md', [self.chunk])

    def test_existing_question_and_answer_preserved(self):
        self.case.update(question=self.generated['question'], expected_answer='Seventeen days.', topic='Existing topic')
        result = merge_case(self.case, self.generated, 'synthetic.md', [self.chunk])
        self.assertEqual(result['expected_answer'], 'Seventeen days.')
        self.assertEqual(result['topic'], 'Existing topic')
        self.generated['question'] = 'Changed question?'
        with self.assertRaises(ValueError):
            merge_case(self.case, self.generated, 'synthetic.md', [self.chunk])

    def test_cross_mentions_excluded(self):
        other = {**self.chunk, 'source': 'other.md'}
        kept = {**self.chunk, 'source': 'safe.md', 'title': 'Unrelated', 'text': 'An unrelated fact.'}
        entities = [{'source': 'synthetic.md', 'aliases': ['Example-42']}]
        self.assertEqual(safe_chunks([self.chunk, other, kept], entities), [kept])

    def test_progress_survives_failure_and_rejects_changed_inputs(self):
        area = Path(__file__).resolve().parent / '.test-work'
        root = area / uuid.uuid4().hex
        root.mkdir(parents=True)
        try:
            path = root / 'progress.json'
            cases = [self.case, {**self.case, 'id': 'known_7'}]
            checkpoint(path, 'synthetic-hash', [self.case])
            self.assertEqual(resume(path, 'synthetic-hash', cases), [self.case])
            with self.assertRaises(SystemExit):
                resume(path, 'changed-hash', cases)
            self.assertEqual(resume(path, 'synthetic-hash', cases), [self.case])
            checkpoint(path, 'synthetic-hash', cases)
            self.assertEqual(len(resume(path, 'synthetic-hash', cases)), 2)
        finally:
            if not root.resolve().is_relative_to(area.resolve()):
                raise ValueError('Unsafe cleanup')
            shutil.rmtree(root)

    def test_error_diagnostics_do_not_echo_model_content(self):
        error = json.JSONDecodeError('Invalid', 'PRIVATE SYNTHETIC TEXT', 0)
        self.assertEqual(failure_reason(error), 'Invalid JSON returned by Ollama')
        self.assertEqual(failure_reason(ValueError('Quote is not an exact source excerpt')),
                         'Quote is not an exact source excerpt')

    def test_quote_id_resolves_to_source_without_model_copy(self):
        session = Mock()
        generated = {k: v for k, v in self.generated.items() if k != 'quote'}
        generated['quote_id'] = 'Q1'
        response = session.post.return_value
        response.status_code = 200
        response.json.return_value = {'done': True, 'message': {'content': json.dumps(generated)}}
        settings = SimpleNamespace(ollama_url='http://127.0.0.1:11434', llm_model='qwen3:8b')
        result = generate(session, settings, {'chunks': [self.chunk]})
        self.assertEqual(result['quote'], self.chunk['text'])
        self.assertNotIn('quote_id', result)
        self.assertFalse(session.post.call_args.kwargs['allow_redirects'])
        self.assertEqual(session.post.call_args.kwargs['json']['format']['properties']['quote_id']['enum'], ['Q1'])
        generated['quote_id'] = 'Q999'
        response.json.return_value = {'done': True, 'message': {'content': json.dumps(generated)}}
        with self.assertRaisesRegex(ValueError, 'Unknown source quote ID'):
            generate(session, settings, {'chunks': [self.chunk]})

    def test_all_quote_options_are_exact_substrings(self):
        chunk = {'text': 'First synthetic sentence.\nSecond synthetic sentence! Third synthetic sentence?'}
        options = quote_options([chunk])
        self.assertEqual(len(options), 3)
        self.assertTrue(all(text in chunk['text'] for text in options.values()))


if __name__ == '__main__':
    unittest.main()
