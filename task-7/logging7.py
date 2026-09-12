"""Log every task 7 query; keep errors distinct from correct knowledge refusals."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from shared7 import BASE, ROOT, LiveRAG, rag, read_snapshot, digest
from golden import validate_golden, score
from settings7 import load_settings
from grounding7 import GroundedClient

VARIANTS = ('baseline', 'gaps', 'restored')


class LoggedRAG:
    def __init__(self, engine, variant, log_path, metadata, cases=()):
        self.engine, self.variant = engine, variant
        self.log_path, self.metadata = Path(log_path), metadata
        self.cases = list(cases)
        self.run_id = uuid.uuid4().hex

    def ask(self, question, case=None):
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = self.engine.ask(question)
        except Exception as exc:
            result = {'question': question, 'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc)}
        raw = result.get('audit', {}).get('retrieved', result.get('retrieved', []))
        final = result.get('retrieved', [])
        answer = result.get('answer', {})
        ids = {e['passage_id'] for e in answer.get('evidence', [])}
        record = {**self.metadata, 'run_id': self.run_id, 'variant': self.variant,
                  'timestamp': timestamp, 'query': question, 'status': result['status'],
                  'found_chunks': bool(raw), 'retrieved_count': len(raw),
                  'found_sources': sorted({p['source'] for p in raw}),
                  'used_sources': sorted({p['source'] for p in final if p['passage_id'] in ids}),
                  'answer': answer, 'reply': result.get('reply', ''),
                  'answer_length': len(answer.get('answer', '')),
                  'success': result['status'] == 'answered' and bool(answer.get('evidence')),
                  'success_basis': 'answer with validated citations; not semantic proof',
                  'seconds': round(time.perf_counter() - started, 3),
                  'few_shot_count': result.get('few_shot_count', 0), 'error': result.get('error')}
        if case is not None:
            record.update({'case_id': case['id'], 'topic': case['topic'], 'kind': case['kind'],
                           **score(case, self.variant, result)})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return {**result, 'log_record': record}


def open_session(variant, settings=None):
    if variant not in VARIANTS:
        raise ValueError('Unknown snapshot variant')
    cases_path, entities_path = BASE / 'golden_questions.json', BASE / 'gap_entities.json'
    cases, entities = json.loads(cases_path.read_text(encoding='utf-8')), json.loads(entities_path.read_text(encoding='utf-8'))
    baseline = BASE / 'index/baseline.zip'
    _, chunks, _ = read_snapshot(baseline)
    validate_golden(cases, entities, chunks)
    _, _, gap_report = read_snapshot(BASE / 'index/gaps.zip')
    if (gap_report['golden_sha256'] != digest(cases_path.read_bytes()) or
            gap_report['entities_sha256'] != digest(entities_path.read_bytes()) or
            gap_report['baseline_sha256'] != digest(baseline.read_bytes())):
        raise ValueError('Experiment configuration changed; recreate variants and rerun all evaluations')
    path = BASE / 'index' / (variant + '.zip')
    if variant == 'restored' and digest(path.read_bytes()) != digest(baseline.read_bytes()):
        raise ValueError('Restored snapshot must match the baseline')
    excluded = {source for c in cases for source in c.get('expected_sources', [])}
    settings = settings or load_settings()
    client = GroundedClient(settings)
    client.check()
    engine = LiveRAG(settings, snapshot_path=path, excluded_sources=excluded, client=client)
    metadata = {'model': engine.settings.llm_model, 'model_digest': engine.client.model_digest,
                'snapshot_sha256': engine.retriever.report['snapshot_sha256'],
                'baseline_sha256': gap_report['baseline_sha256'],
                'golden_sha256': gap_report['golden_sha256'], 'entities_sha256': gap_report['entities_sha256'],
                'implementation_sha256': digest(b''.join((ROOT / name).read_bytes() for name in (
                    'task-4/rag.py', 'task-4/prompts.py', 'task-5/pipeline.py', 'task-5/security.py',
                    'task-6/runtime6.py', 'task-7/golden.py', 'task-7/grounding7.py', 'task-7/logging7.py'))),
                'temperature': 0, 'seed': 7, 'retrieval_candidates': 12, 'unique_documents': 3}
    return LoggedRAG(engine, variant, BASE / 'logs.jsonl', metadata, cases)
