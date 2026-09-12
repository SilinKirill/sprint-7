"""Evaluate a user-approved golden set against one frozen snapshot."""
import argparse
import json
from shared7 import BASE
from logging7 import open_session, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    session = open_session(args.variant)
    directory = BASE / 'results'
    directory.mkdir(exist_ok=True)
    rows = []
    for case in session.cases:
        row = session.ask(case['question'], case)['log_record']
        rows.append(row)
        print(f"{case['id']}: expected={row['expected_status']}, actual={row['status']}, "
              f"auto_pass={row['auto_pass']}, completeness={row['keyword_completeness']}", flush=True)
        payload = {'run_id': session.run_id, 'variant': args.variant,
                   'complete': len(rows) == len(session.cases), 'runs': rows}
        (directory / (args.variant + '.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    # A new run requires fresh human review; never reuse approval for older answers.
    (directory / ('review_' + args.variant + '.json')).write_text(json.dumps({
        'run_id': session.run_id, 'judgments': {r['case_id']: None for r in rows},
        'notes': 'Replace null with true/false after comparing the answer to the expected facts.'}, indent=2), encoding='utf-8')
    passed = sum(r['auto_pass'] for r in rows)
    print(f'{args.variant}: automatic checks {passed}/{len(rows)}. Review answers locally before drawing conclusions.')
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
