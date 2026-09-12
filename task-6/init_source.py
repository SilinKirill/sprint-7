"""Run locally once to create task 6's editable source copy."""
import shutil
from shared6 import BASE, ROOT


def main():
    source = ROOT / 'task-2/knowledge_base'
    target = BASE / 'incoming'
    if target.exists():
        raise SystemExit('task-6/incoming already exists; preserved.')
    files = sorted(source.glob('*.md'))
    if len(files) < 30 or any(p.is_symlink() for p in files):
        raise ValueError('Expected at least 30 ordinary Markdown documents')
    target.mkdir()
    for path in files:
        shutil.copyfile(path, target / path.name)
    print(f'Prepared task-6/incoming: {len(files)} files. Originals unchanged.')


if __name__ == '__main__':
    main()
