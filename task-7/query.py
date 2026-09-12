"""Ask and log an independent question against a selected experimental snapshot."""
import argparse
from logging7 import open_session, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default='gaps')
    parser.add_argument('--query', required=True)
    args = parser.parse_args()
    result = open_session(args.variant).ask(args.query)
    print(result.get('reply', result.get('error', 'No response')))
    if result['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
