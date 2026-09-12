"""Create/change/remove one owned synthetic document for a real update demo."""
import argparse
import json
import secrets
from shared6 import BASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('add', 'change', 'remove'))
    args = parser.parse_args()
    path, state = BASE / 'incoming/update_demo.md', BASE / 'demo_state.json'
    if not path.parent.is_dir():
        raise SystemExit('Run init_source.py first')
    if args.action == 'add':
        if path.exists() or state.exists():
            raise SystemExit('Demo already exists; use change or remove.')
        info = {'name': 'Maintenance unit DEMO-' + secrets.token_hex(4), 'interval_days': 17}
    else:
        info = json.loads(state.read_text(encoding='utf-8'))
        if not path.is_file() or not path.read_text(encoding='utf-8').startswith('# ' + info['name'] + '\n'):
            raise SystemExit('Demo file ownership check failed; no changes made.')
        if args.action == 'remove':
            path.unlink()
            state.unlink()
            print('Removed only incoming/update_demo.md. Run update_index.py.')
            return
        info['interval_days'] = 29 if info['interval_days'] == 17 else 17
    text = (f"# {info['name']}\n\nThis unit inspects warehouse calibration equipment. "
            f"Its mandatory calibration interval is exactly {info['interval_days']} days. "
            'It records each inspection in a local register and must remain stationary during calibration.\n')
    path.write_text(text, encoding='utf-8')
    state.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f"Question: What is the calibration interval of {info['name']}?")
    print('Expected days:', info['interval_days'])
    print('Run update_index.py, then query.py with the question above.')


if __name__ == '__main__':
    main()
