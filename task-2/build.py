"""Apply the saved dictionary, or check the existing result without writing."""
import argparse
import json
import re
from pathlib import Path

from generate_terms import load_catalog, alias_index, source_pattern, validate_map

BASE = Path(__file__).resolve().parent


def replace(text, mapping, catalog):
    pattern = source_pattern(catalog)
    pattern = re.compile(pattern.pattern + r"(?P<poss>['’]s|['’](?!\w))?", pattern.flags)
    lookup = {key.casefold(): value for key, value in mapping.items()}
    def substitute(match):
        suffix = match.group('poss') or ''
        original = match.group()[:-len(suffix)] if suffix else match.group()
        value = lookup[original.casefold()]
        if original[0].isupper() and value[0].islower():
            value = value[0].upper() + value[1:]
        if suffix in ("'", '’'):
            suffix += '' if value.lower().endswith('s') else 's'
        return value + suffix
    return pattern.subn(substitute, text)


def load_prepared(catalog):
    sources = json.loads((BASE / 'sources.json').read_text(encoding='utf-8'))
    if len(sources) != 30 or len({s['pageid'] for s in sources}) != 30:
        raise ValueError('Expected 30 distinct sources')
    expected = {s['file'] for s in sources}
    if any(not re.fullmatch(r'doc_\d{2}\.md', f) for f in expected) or len(expected) != 30:
        raise ValueError('Invalid source filenames')
    if {p.name for p in (BASE / 'prepared').iterdir()} != expected:
        raise ValueError('prepared/ must contain exactly the 30 listed documents')
    pattern, owners = source_pattern(catalog), alias_index(catalog)
    texts = {}
    for source in sources:
        text = (BASE / 'prepared' / source['file']).read_text(encoding='utf-8')
        heading, body = text.split('\n\n', 1)
        if heading != '# ' + source['title'].removesuffix('/Legends'):
            raise ValueError('Heading mismatch: ' + source['file'])
        words = len(re.findall(r"\b[\w]+(?:[-'’][\w]+)*\b", body))
        if not 150 <= words <= 400:
            raise ValueError('Unexpected length: ' + source['file'])
        matches = list(pattern.finditer(body))
        if len({owners[m.group().casefold()] for m in matches}) < 3:
            raise ValueError('Need multiple entities in body: ' + source['file'])
        sentences = re.split(r'(?<=[.!?])\s+', body)
        if sum(bool(pattern.search(s)) for s in sentences) < 2:
            raise ValueError('Need matches in multiple sentences: ' + source['file'])
        if re.search(r'https?://|<[^>]+>|\[\d+\]|Behind the scenes|Notes and references|/Legends', text):
            raise ValueError('Source debris: ' + source['file'])
        texts[source['file']] = text
    if len(set(texts.values())) != 30:
        raise ValueError('Duplicate documents')
    return texts


def expected_outputs(texts, mapping, catalog):
    outputs, count = {}, 0
    remaining = source_pattern(catalog)
    for filename, text in texts.items():
        output, changed = replace(text, mapping, catalog)
        if remaining.search(output):
            raise ValueError('Original term remains: ' + filename)
        body = text.split('\n\n', 1)[1]
        if replace(body, mapping, catalog)[1] < 2:
            raise ValueError('No meaningful body replacements: ' + filename)
        outputs[filename] = output
        count += changed
    if len(set(outputs.values())) != 30:
        raise ValueError('Duplicate output documents')
    return outputs, count


def check_files(outputs):
    folder = BASE / 'knowledge_base'
    if not folder.is_dir() or {p.name for p in folder.iterdir()} != set(outputs):
        raise ValueError('knowledge_base/ must contain exactly the expected 30 files')
    for name, expected in outputs.items():
        if (folder / name).read_text(encoding='utf-8') != expected:
            raise ValueError('Result differs from saved dictionary + prepared text: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    catalog = load_catalog()
    path = BASE / 'terms_map.json'
    if not path.exists():
        raise SystemExit('Run generate_terms.py first; no dictionary exists.')
    mapping = json.loads(path.read_text(encoding='utf-8'))
    validate_map(mapping, catalog)
    outputs, count = expected_outputs(load_prepared(catalog), mapping, catalog)
    folder = BASE / 'knowledge_base'
    if args.check or folder.exists():
        check_files(outputs)
        print(f'Checks passed: 30 documents; {count} replacements. Nothing written.')
    else:
        folder.mkdir()
        for name, text in outputs.items():
            (folder / name).write_text(text, encoding='utf-8')
        check_files(outputs)
        print(f'Built: 30 documents; {count} replacements.')
    print('Checks cover the listed terms and exact transformation, not guaranteed plot concealment.')


if __name__ == '__main__':
    main()
