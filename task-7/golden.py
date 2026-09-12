"""Validate locally supplied ground truth before any model evaluation."""
import json
import re
import unicodedata


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def contains(text, term):
    return re.search(r'(?<!\w)' + re.escape(norm(term)) + r'(?!\w)', norm(text)) is not None


def supported(reference, chunks):
    return any(c['source'] == reference['source'] and norm(reference['text']) in norm(c['text'])
               for c in chunks)


def validate_golden(cases, entities, chunks):
    ids = [c.get('id') for c in cases]
    if not 10 <= len(cases) <= 15 or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError('Use 10-15 cases with unique nonempty IDs')
    known = [c for c in cases if c.get('kind') == 'known']
    missing = [c for c in cases if c.get('kind') in ('gap', 'absent')]
    if not 6 <= len(known) <= 8 or not 3 <= len(missing) <= 5 or len(known) + len(missing) != len(cases):
        raise ValueError('Use 6-8 known cases and 3-5 gap/absent cases')
    if len({norm(c.get('question', '')) for c in cases}) != len(cases):
        raise ValueError('Questions must be distinct')
    if len(entities) not in (2, 3) or len({e.get('id') for e in entities}) != len(entities):
        raise ValueError('Select 2-3 entities with unique IDs')
    sources = {c['source'] for c in chunks}
    entity_map = {e['id']: e for e in entities}
    for e in entities:
        if e.get('source') not in sources or not e.get('aliases') or any(not a.strip() for a in e['aliases']):
            raise ValueError(f"Fill source and every known alias for {e.get('id')}")
        own = [c for c in chunks if c['source'] == e['source']]
        if not any(contains(c['title'] + '\n' + c['text'], a) for c in own for a in e['aliases']):
            raise ValueError(f"No selected alias occurs in source for {e['id']}")
    if len({e['source'] for e in entities}) != len(entities):
        raise ValueError('Each removed entity must have a different primary document')
    gap_ids = {c.get('entity_id') for c in cases if c['kind'] == 'gap'}
    if gap_ids != set(entity_map):
        raise ValueError('Provide gap questions covering each selected entity')
    for c in cases:
        if not c.get('question', '').strip() or not c.get('topic', '').strip():
            raise ValueError(f"Fill question and topic for {c['id']}")
        if c['kind'] == 'absent':
            continue
        keywords = c.get('answer_keywords', [])
        refs = c.get('reference_quotes', [])
        expected_sources = c.get('expected_sources', [])
        if (not c.get('expected_answer', '').strip() or not keywords or
                any(not k.strip() or not contains(c['expected_answer'], k) for k in keywords) or
                not expected_sources or not set(expected_sources) <= sources or not refs):
            raise ValueError(f"Fill expected answer, keywords, sources and exact reference quotes for {c['id']}")
        for ref in refs:
            if len(norm(ref.get('text', ''))) < 12 or ref.get('source') not in expected_sources or not supported(ref, chunks):
                raise ValueError(f"Reference quote does not match baseline for {c['id']}")
        if c['kind'] == 'gap':
            if entity_map[c['entity_id']]['source'] not in expected_sources:
                raise ValueError(f"Gap case {c['id']} must cite its entity's primary source")
    return cases


def score(case, variant, result):
    expected = 'unknown' if case['kind'] == 'absent' or (case['kind'] == 'gap' and variant == 'gaps') else 'answered'
    answer = result.get('answer', {})
    text = answer.get('answer', '')
    chosen = {e['passage_id'] for e in answer.get('evidence', [])}
    cited = {p['source'] for p in result.get('retrieved', []) if p['passage_id'] in chosen}
    raw = result.get('audit', {}).get('retrieved', result.get('retrieved', []))
    expected_sources = set(case.get('expected_sources', []))
    keywords = case.get('answer_keywords', [])
    completeness = sum(contains(text, k) for k in keywords) / len(keywords) if keywords else None
    source_ok = bool(cited & expected_sources) if expected_sources else None
    passed = (result['status'] == 'answered' and source_ok and completeness == 1) if expected == 'answered' else (
        result['status'] == 'unknown' and not answer.get('answerable', False))
    return {'expected_status': expected, 'auto_pass': bool(passed),
            'keyword_completeness': completeness, 'expected_source_cited': source_ok,
            'retrieval_hit': bool({p['source'] for p in raw} & expected_sources) if expected_sources else None}
