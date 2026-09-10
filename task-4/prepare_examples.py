"""Create two extractive demonstrations locally; no invented facts or LLM call."""
import argparse
import json
import re

from rag import BASE, load_task3_index, validate_answer


def make_examples(chunks, report, excluded_sources):
    examples, used = [], set()
    for chunk in chunks:
        if chunk['source'] in used or chunk['source'] in excluded_sources:
            continue
        # Ignore headings/epigraphs; use an actual descriptive paragraph.
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', chunk['text'])
                      if p.strip() and not p.lstrip().startswith(('#', '>'))]
        paragraph = next((p for p in paragraphs if len(p.split()) >= 20), None)
        if paragraph is None:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        quote = sentences[0]
        if len(quote) < 40 and len(sentences) > 1:
            quote = paragraph[:paragraph.index(sentences[1]) + len(sentences[1])]
        if not 12 <= len(quote) <= 1000:
            continue
        p = {**chunk, 'passage_id': f'EXAMPLE{len(examples)+1}', 'text': quote}
        response = {'answerable': True, 'answer': 'The passage states: ' + quote,
                    'explanation': ['The passage describes the specified entity.',
                                    'The answer reproduces information from that description.'],
                    'evidence': [{'passage_id': p['passage_id'], 'quote': quote}]}
        validate_answer(json.dumps(response), [p])
        examples.append({'question': f'What does the passage say about {chunk["title"]}?',
                         'passage': p, 'response': response})
        used.add(chunk['source'])
        if len(examples) == 2:
            break
    if len(examples) != 2:
        raise ValueError('Could not select two examples from separate documents')
    return {'index_sha256': report['artifact_sha256']['faiss.index'], 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    path = BASE / 'few_shot.json'
    if path.exists() and not args.overwrite:
        raise SystemExit('few_shot.json already exists; preserved. Use --overwrite only to regenerate.')
    _, chunks, report = load_task3_index()
    cases = json.loads((BASE / 'questions.json').read_text(encoding='utf-8'))
    excluded = {case['expected_source'] for case in cases if case.get('expected_source')}
    result = make_examples(chunks, report, excluded)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved task-4/few_shot.json: 2 source-based examples, outside the five test source documents.')
    print('Review the examples locally before running the bot.')


if __name__ == '__main__':
    main()
