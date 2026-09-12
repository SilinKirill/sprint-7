"""Run five expected answers and two expected refusals; save actual local results."""
from datetime import datetime, timezone
import json

from rag import BASE, RAG, Settings, format_reply, ServiceError, OutputError, RETRIEVAL_CANDIDATES


def main():
    engine = RAG(Settings.from_env())
    cases = json.loads((BASE / 'questions.json').read_text(encoding='utf-8'))
    results = []
    for case in cases:
        try:
            result = engine.ask(case['question'])
            result['status_matches'] = result['status'] == case['expected_status']
            expected = case.get('expected_source')
            used_ids = {e['passage_id'] for e in result['answer']['evidence']}
            cited = {p['source'] for p in result['retrieved'] if p['passage_id'] in used_ids}
            result['expected_source_cited'] = expected in cited if expected else None
            result['reply'] = format_reply(result)
        except (ValueError, ServiceError, OutputError) as exc:
            result = {'question': case['question'], 'status': 'error',
                      'status_matches': False, 'error': str(exc)}
        result['case_id'] = case['id']
        result['expected_status'] = case['expected_status']
        results.append(result)
        print(f'\n{case["id"]}: expected={case["expected_status"]}, actual={result["status"]}', flush=True)
        print(result.get('reply', result.get('error', '')), flush=True)
        # Keep completed results even if a later request is interrupted.
        report = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'llm_model': engine.settings.llm_model,
                  'llm_digest': engine.client.model_digest,
                  'embedding_model': engine.retriever.report['model'],
                  'index_sha256': engine.retriever.report['artifact_sha256']['faiss.index'],
                  'settings': {'temperature': 0, 'seed': 7, 'top_k': 3,
                               'candidate_k': RETRIEVAL_CANDIDATES, 'unique_sources': True,
                               'few_shot_examples': 2, 'explanation': 'short evidence-based steps'},
                  'complete': len(results) == len(cases), 'dialogues': results}
        (BASE / 'dialogues.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    matching = sum(r['status_matches'] for r in results)
    cited = sum(r.get('expected_source_cited') is True for r in results)
    print(f'\nSaved task-4/dialogues.json. Status matches: {matching}/{len(cases)}; expected sources cited: {cited}/5.')
    print('Check factual correctness locally. Matching statuses and citations alone do not prove correctness.')
    if matching != len(cases) or cited != 5:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
