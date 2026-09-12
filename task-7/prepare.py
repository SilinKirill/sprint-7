"""Freeze a clean baseline; create gaps by removing entities and mentioning chunks."""
import argparse
import json
from pathlib import Path
import shutil

import faiss
import numpy as np
from shared7 import BASE, ROOT, read_snapshot, write_snapshot, digest, DIMENSION
from golden import contains, supported, validate_golden


def draft_case(number, kind):
    return {'id': f'{kind}_{number}', 'kind': kind, 'topic': '', 'question': '',
            'expected_answer': '', 'answer_keywords': [], 'expected_sources': [],
            'reference_quotes': [{'source': '', 'text': ''}] if kind != 'absent' else []}


def remove_entities(index, chunks, entities):
    primary = {e['source'] for e in entities}
    aliases = [a for e in entities for a in e['aliases']]
    kept, removed, positions = [], [], []
    for i, c in enumerate(chunks):
        if c['source'] in primary or any(contains(c['title'] + '\n' + c['text'], a) for a in aliases):
            removed.append(c['chunk_id'])
        else:
            kept.append(c)
            positions.append(i)
    result = faiss.IndexFlatIP(DIMENSION)
    if positions:
        old = index.reconstruct_n(0, index.ntotal)
        result.add(np.ascontiguousarray(old[positions], dtype='float32'))
    return result, kept, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('init', 'gaps'), required=True)
    args = parser.parse_args()
    directory = BASE / 'index'
    baseline = directory / 'baseline.zip'
    if args.stage == 'init':
        if baseline.exists() or (BASE / 'golden_questions.json').exists():
            raise SystemExit('Experiment already initialized; files preserved.')
        source = ROOT / 'task-6/index/snapshot.zip'
        _, chunks, report = read_snapshot(source)
        directory.mkdir(exist_ok=True)
        shutil.copyfile(source, baseline)
        docs = {}
        for c in chunks:
            docs.setdefault(c['source'], {'source': c['source'], 'title': c['title'], 'chunks': []})['chunks'].append(c)
        # This helper file stays local to the user's preparation workflow.
        (BASE / 'source_catalog.json').write_text(json.dumps(list(docs.values()), ensure_ascii=False, indent=2), encoding='utf-8')
        cases = [draft_case(i, 'known') for i in range(1, 9)]
        earlier = json.loads((ROOT / 'task-4/questions.json').read_text(encoding='utf-8'))
        for case, old in zip(cases[:5], earlier[:5]):
            matches = [s for s in docs if Path(s).name == Path(old['expected_source']).name]
            case['question'] = old['question']
            if len(matches) == 1:
                case['expected_sources'] = matches
                case['reference_quotes'][0]['source'] = matches[0]
        for i in range(1, 4):
            cases.append({**draft_case(i, 'gap'), 'entity_id': f'E{i}'})
        cases.append({**draft_case(1, 'absent'), 'topic': 'Live weather',
                      'question': 'What is the current air temperature in Paris?'})
        (BASE / 'golden_questions.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
        (BASE / 'gap_entities.json').write_text(json.dumps([
            {'id': f'E{i}', 'source': '', 'aliases': []} for i in range(1, 4)], indent=2), encoding='utf-8')
        print(f'Baseline frozen: {len(chunks)} chunks. Fill golden_questions.json and gap_entities.json locally.')
        print('source_catalog.json contains titles and chunk excerpts for local reference selection.')
        return
    index, chunks, report = read_snapshot(baseline)
    cases = json.loads((BASE / 'golden_questions.json').read_text(encoding='utf-8'))
    entities = json.loads((BASE / 'gap_entities.json').read_text(encoding='utf-8'))
    validate_golden(cases, entities, chunks)
    reduced, retained, removed = remove_entities(index, chunks, entities)
    for c in cases:
        if c['kind'] == 'known' and any(not supported(ref, retained) for ref in c['reference_quotes']):
            raise ValueError(f"Gaps also remove reference facts for {c['id']}; choose more independent entities/questions before testing.")
    if not removed:
        raise ValueError('No chunks removed')
    if (directory / 'gaps.zip').exists() or (directory / 'restored.zip').exists():
        raise SystemExit('Variant snapshots already exist; preserved. See README to rebuild the experiment.')
    metadata = {**report, 'variant': 'gaps', 'removed_chunks': removed,
                'baseline_sha256': digest(baseline.read_bytes()),
                'golden_sha256': digest((BASE / 'golden_questions.json').read_bytes()),
                'entities_sha256': digest((BASE / 'gap_entities.json').read_bytes()),
                'documents': len({c['source'] for c in retained})}
    metadata.pop('snapshot_sha256', None)
    metadata.pop('files', None)  # Source-file hashes would misrepresent a filtered snapshot.
    write_snapshot(directory / 'gaps.zip', reduced, retained, metadata)
    shutil.copyfile(baseline, directory / 'restored.zip')
    exported = '\n\n'.join(f"{c['id']} [{c['kind']}] {c['question']}\nExpected: " +
                            (c['expected_answer'] if c['kind'] != 'absent' else "I don't know") +
                            ('\nIn gaps: I don\'t know' if c['kind'] == 'gap' else '') for c in cases)
    (BASE / 'golden_questions.txt').write_text(exported + '\n', encoding='utf-8')
    print(f'Prepared gaps: {len(removed)} chunks removed, {len(retained)} remain. Restored copy prepared.')
    print('Review gap_entities aliases for completeness; unknown aliases or paraphrases may remain.')


if __name__ == '__main__':
    main()
