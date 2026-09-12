"""Synthetic golden sets, FAISS vectors, snapshots and request logs only."""
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import Mock, patch

import faiss
import numpy as np
from shared7 import DIMENSION
from golden import contains, validate_golden, score
from prepare import remove_entities
from logging7 import LoggedRAG
from analyze import summarize
import prepare
import analyze
from shared7 import write_snapshot, read_snapshot, digest
from shared6 import MODEL, QUERY_PROMPT


def dataset():
    chunks, cases, entities = [], [], []
    for i in range(1, 12):
        text = f'Entity{i} uses a silver battery. Its maintenance takes place inside a research station where operators carefully record the inspection results every day.'
        chunk = {'chunk_id': f'doc{i}:0', 'source': f'fake/doc{i}.md', 'title': f'Entity{i}',
                 'text': text, 'passage_id': 'C1', 'start_line': 1, 'end_line': 2}
        chunks.append(chunk)
        case = {'id': f'Q{i}', 'kind': 'known' if i <= 8 else 'gap', 'question': f'What powers Entity{i}?',
                'topic': f'Topic{i}', 'expected_answer': f'Entity{i} uses a silver battery.',
                'answer_keywords': [f'Entity{i}', 'silver battery'], 'expected_sources': [chunk['source']],
                'reference_quotes': [{'source': chunk['source'], 'text': text}]}
        if i > 8:
            case['entity_id'] = f'E{i}'
            entities.append({'id': f'E{i}', 'source': chunk['source'], 'aliases': [f'Entity{i}']})
        cases.append(case)
    cases.append({'id': 'Q12', 'kind': 'absent', 'question': 'What is the weather now?', 'topic': 'Weather'})
    index = faiss.IndexFlatIP(DIMENSION)
    vectors = np.eye(DIMENSION, dtype='float32')[:len(chunks)]
    index.add(vectors)
    return index, chunks, cases, entities


def answer_result(chunk, text=None):
    return {'status': 'answered', 'answer': {'answerable': True, 'answer': text or chunk['title'] + ' uses a silver battery.',
            'explanation': ['The text specifies the supply.', 'The supply is a silver battery.'],
            'evidence': [{'passage_id': 'C1', 'quote': chunk['text']}]},
            'retrieved': [chunk], 'reply': 'Synthetic answer', 'few_shot_count': 2}


class EvaluationTests(unittest.TestCase):
    def test_prepare_variants_and_analyze_completed_runs(self):
        area = Path(__file__).resolve().parent / '.test-work'
        directory = area / uuid.uuid4().hex
        root = directory / 'project'
        target = root / 'task-7'
        target.mkdir(parents=True)
        try:
            index, chunks, cases, entities = dataset()
            report = {'model': MODEL, 'dimension': DIMENSION, 'query_prompt': QUERY_PROMPT,
                      'model_revision': 'synthetic-revision', 'documents': 11}
            write_snapshot(root / 'task-6/index/snapshot.zip', index, chunks, report)
            (root / 'task-4').mkdir()
            (root / 'task-4/questions.json').write_text(json.dumps([
                {'question': c['question'], 'expected_source': c['expected_sources'][0]} for c in cases[:5]]), encoding='utf-8')
            with patch.object(prepare, 'ROOT', root), patch.object(prepare, 'BASE', target):
                with patch('sys.argv', ['prepare.py', '--stage', 'init']):
                    prepare.main()
                (target / 'golden_questions.json').write_text(json.dumps(cases), encoding='utf-8')
                (target / 'gap_entities.json').write_text(json.dumps(entities), encoding='utf-8')
                with patch('sys.argv', ['prepare.py', '--stage', 'gaps']):
                    prepare.main()
            self.assertEqual(read_snapshot(target / 'index/gaps.zip')[0].ntotal, 8)
            self.assertEqual((target / 'index/baseline.zip').read_bytes(), (target / 'index/restored.zip').read_bytes())
            self.assertTrue((target / 'golden_questions.txt').exists())
            (target / 'results').mkdir()
            hashes = {'baseline_sha256': digest((target / 'index/baseline.zip').read_bytes()),
                      'golden_sha256': digest((target / 'golden_questions.json').read_bytes()),
                      'entities_sha256': digest((target / 'gap_entities.json').read_bytes()),
                      'model': 'synthetic', 'model_digest': 'synthetic', 'implementation_sha256': 'synthetic'}
            for variant in ('baseline', 'gaps', 'restored'):
                rows = []
                for i, case in enumerate(cases):
                    result = {'status': 'unknown', 'answer': {'answerable': False}, 'retrieved': []}
                    if case['kind'] != 'absent' and not (variant == 'gaps' and case['kind'] == 'gap'):
                        result = answer_result(chunks[i])
                    rows.append({**hashes, 'case_id': case['id'], 'status': result['status'], **score(case, variant, result)})
                data = {'complete': True, 'run_id': variant, 'runs': rows}
                (target / 'results' / (variant + '.json')).write_text(json.dumps(data), encoding='utf-8')
                (target / 'results' / ('review_' + variant + '.json')).write_text(json.dumps({
                    'run_id': variant, 'judgments': {c['id']: True for c in cases}}), encoding='utf-8')
            with patch.object(analyze, 'BASE', target):
                analyze.main()
            summary = json.loads((target / 'results/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['variants']['gaps']['automatic_pass'], 12)
            self.assertEqual(summary['variants']['baseline']['unknown'], 1)
            self.assertEqual(summary['variants']['gaps']['unknown'], 4)
            self.assertTrue((target / 'REPORT.md').exists())
        finally:
            if not directory.resolve().is_relative_to(area.resolve()):
                raise ValueError('Unsafe test cleanup path')
            shutil.rmtree(directory)

    def test_valid_set_and_unfilled_template(self):
        _, chunks, cases, entities = dataset()
        validate_golden(cases, entities, chunks)
        cases[0]['expected_answer'] = ''
        with self.assertRaises(ValueError):
            validate_golden(cases, entities, chunks)

    def test_reference_must_exist_in_baseline(self):
        _, chunks, cases, entities = dataset()
        cases[0]['reference_quotes'][0]['text'] = 'An invented fact which never appears in the source.'
        with self.assertRaises(ValueError):
            validate_golden(cases, entities, chunks)

    def test_aliases_remove_cross_document_mentions(self):
        index, chunks, _, entities = dataset()
        cross = {**chunks[0], 'chunk_id': 'cross', 'source': 'fake/cross.md',
                 'title': 'Cross reference', 'text': 'This external unit works with Entity9.'}
        unaffected = {**cross, 'chunk_id': 'other', 'source': 'fake/other.md', 'text': 'Entity90 is a distinct identifier.'}
        index.add(np.eye(DIMENSION, dtype='float32')[11:13])
        reduced, retained, removed = remove_entities(index, chunks + [cross, unaffected], entities)
        self.assertEqual(len(removed), 4)
        self.assertEqual(reduced.ntotal, 9)
        self.assertIn('other', {c['chunk_id'] for c in retained})
        self.assertFalse(any(contains(c['title'] + c['text'], a) for c in retained for e in entities for a in e['aliases']))
        np.testing.assert_array_equal(reduced.reconstruct(8), np.eye(DIMENSION, dtype='float32')[12])

    def test_keyword_boundaries(self):
        self.assertFalse(contains('Entity90', 'Entity9'))
        self.assertTrue(contains("ENTITY9's battery", 'entity9'))

    def test_answer_and_refusal_expectations_change_by_variant(self):
        _, chunks, cases, _ = dataset()
        case, chunk = cases[8], chunks[8]
        result = answer_result(chunk)
        self.assertTrue(score(case, 'baseline', result)['auto_pass'])
        self.assertFalse(score(case, 'gaps', result)['auto_pass'])
        self.assertTrue(score(case, 'restored', result)['auto_pass'])
        unknown = {'status': 'unknown', 'answer': {'answerable': False, 'answer': "I don't know"}, 'retrieved': []}
        self.assertTrue(score(case, 'gaps', unknown)['auto_pass'])
        self.assertFalse(score(case, 'baseline', unknown)['auto_pass'])
        self.assertFalse(score(case, 'gaps', {'status': 'error'})['auto_pass'])

    def test_source_match_alone_does_not_pass(self):
        _, chunks, cases, _ = dataset()
        result = answer_result(chunks[0], 'Entity1 uses a gold battery.')
        check = score(cases[0], 'baseline', result)
        self.assertTrue(check['expected_source_cited'])
        self.assertEqual(check['keyword_completeness'], 0.5)
        self.assertFalse(check['auto_pass'])

    def test_every_request_including_error_is_logged(self):
        area = Path(__file__).resolve().parent / '.test-work'
        directory = area / uuid.uuid4().hex
        directory.mkdir(parents=True)
        try:
            _, chunks, cases, _ = dataset()
            engine = Mock()
            engine.ask.side_effect = [answer_result(chunks[0]), RuntimeError('synthetic outage')]
            logger = LoggedRAG(engine, 'baseline', directory / 'logs.jsonl', {})
            logger.ask(cases[0]['question'], cases[0])
            logger.ask('Another question')
            rows = [json.loads(line) for line in (directory / 'logs.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]['found_chunks'])
            self.assertTrue(rows[0]['auto_pass'])
            self.assertEqual(rows[1]['status'], 'error')
            self.assertFalse(rows[1]['success'])
            self.assertEqual(rows[1]['answer_length'], 0)
            self.assertIn('timestamp', rows[1])
        finally:
            if not directory.resolve().is_relative_to(area.resolve()):
                raise ValueError('Unsafe test cleanup path')
            shutil.rmtree(directory)

    def test_error_not_counted_as_retrieval_miss(self):
        summary = summarize([{'case_id': 'X', 'status': 'error', 'auto_pass': False,
                              'expected_status': 'answered', 'retrieval_hit': False}])
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['retrieval_misses'], [])


if __name__ == '__main__':
    unittest.main()
