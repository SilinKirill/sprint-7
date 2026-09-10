"""Generate a local deterministic dictionary from reviewed entities, never overwrite it."""
import argparse
import json
import random
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load_catalog():
    return json.loads((BASE / 'terms_source.json').read_text(encoding='utf-8'))


def alias_index(catalog):
    result = {}
    ids = set()
    for row in catalog:
        if row['id'] in ids:
            raise ValueError('Duplicate entity ID: ' + row['id'])
        ids.add(row['id'])
        for alias in row['aliases']:
            key = alias.casefold()
            if not alias or key in result:
                raise ValueError('Duplicate/empty alias: ' + alias)
            result[key] = row['id']
    return result


def source_pattern(catalog):
    parts = []
    aliases = [(alias, alias in row.get('case_sensitive', []))
               for row in catalog for alias in row['aliases']]
    for alias, sensitive in sorted(aliases, key=lambda x: (-len(x[0]), x[0])):
        term = re.escape(alias)
        parts.append('(?-i:' + term + ')' if sensitive else term)
    # Hyphens are allowed after a known term only for an explicit listed alias.
    return re.compile(r'(?<![\w-])(?:' + '|'.join(parts) + r')(?![\w-])', re.I)


def validate_map(mapping, catalog):
    owners = alias_index(catalog)
    expected = {alias for row in catalog for alias in row['aliases']}
    if set(mapping) != expected:
        raise ValueError('Dictionary keys must match terms_source.json exactly')
    pattern = source_pattern(catalog)
    occupied = {}
    for original, target in mapping.items():
        if not isinstance(target, str) or not target.strip() or '\n' in target:
            raise ValueError('Invalid target for ' + original)
        if pattern.search(target):
            raise ValueError('Target contains an original term for ' + original)
        key, owner = target.casefold(), owners[original.casefold()]
        if key in occupied and occupied[key] != owner:
            raise ValueError('Target collision between entities: ' + original)
        occupied[key] = owner


def generate(catalog, seed):
    alias_index(catalog)
    rng = random.Random(seed)
    used = set()
    starts = ['Bel', 'Cor', 'Dar', 'El', 'Fen', 'Gal', 'Hel', 'Ith', 'Kel',
              'Lor', 'Mer', 'Nel', 'Or', 'Pel', 'Quel', 'Ral', 'Sel', 'Tal', 'Val', 'Zel']
    middles = ['a', 'e', 'i', 'o', 'u', 'ae', 'ia', 'eo']
    ends = ['dan', 'len', 'mir', 'nor', 'ran', 'sil', 'tor', 'ven', 'ris', 'mon', 'vek', 'lin']

    def word():
        for _ in range(10000):
            value = rng.choice(starts) + rng.choice(middles) + rng.choice(ends)
            if value.casefold() not in used:
                used.add(value.casefold())
                return value
        raise ValueError('Name pool exhausted')

    raw = {}
    for row in catalog:
        root, first, last = word(), word(), word()
        kind = row['type']
        name = first + ' ' + last if kind == 'person' else root
        if kind in ('droid', 'model'):
            name = root + '-' + str(rng.randrange(10, 100))
        if kind == 'ship':
            name = root + ' ' + rng.choice(['Dawn', 'Horizon', 'Promise', 'Voyager', 'Crest'])
        if kind == 'species':
            name = root.lower() + 'an'
        raw[row['id']] = {'root': root, 'first': first, 'last': last, 'name': name}

    rows = {row['id']: row for row in catalog}
    resolved, active = {}, set()

    def resolve(ident):
        if ident in resolved:
            return resolved[ident]
        if ident in active:
            raise ValueError('Cyclic name template: ' + ident)
        active.add(ident)
        fields = dict(raw[ident])

        def token(match):
            key = match.group(1)
            if '.' in key:
                other, field = key.split('.', 1)
                return resolve(other)[field]
            return fields[key]

        fields['name'] = re.sub(r'\{([^{}]+)\}', token, rows[ident].get('name_template', '{name}'))
        # Shared surnames are reflected in short mentions as well as full names.
        if rows[ident]['type'] == 'person':
            fields['first'], fields['last'] = fields['name'].split(' ', 1)
        resolved[ident] = fields
        active.remove(ident)
        return fields

    result = {}
    for row in catalog:
        fields = resolve(row['id'])
        for alias, form in row['aliases'].items():
            result[alias] = form.format(**fields)
    validate_map(result, catalog)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    path = BASE / 'terms_map.json'
    if path.exists():
        raise SystemExit('terms_map.json already exists; preserved. Run build.py to reuse it.')
    mapping = generate(load_catalog(), args.seed)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(mapping, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'Saved terms_map.json: {len(mapping)} aliases. Seed: {args.seed}.')


if __name__ == '__main__':
    main()
