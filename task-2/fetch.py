"""Download pinned Wookieepedia revisions and extract complete in-universe prose."""
import gzip
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, NavigableString, Comment

BASE = Path(__file__).resolve().parent
API = 'https://starwars.fandom.com/api.php'
STOP = {'behind the scenes', 'appearances', 'sources', 'notes and references',
        'external links', 'bibliography', 'non-canon appearances', 'references'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_cache():
    path = BASE / 'sources_cache.json.gz'
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}


def save_cache(cache):
    (BASE / 'sources_cache.json.gz').write_bytes(
        gzip.compress(json.dumps(cache, ensure_ascii=False).encode('utf-8'), mtime=0))


def download(title, revision=None):
    params = {'action': 'parse', 'format': 'json', 'prop': 'text|revid', 'redirects': 1}
    params.update({'oldid': revision} if revision else {'page': title})
    response = requests.get(API, params=params, timeout=45,
                            headers={'User-Agent': 'Sprint7Study/3.0 (educational corpus)'})
    response.raise_for_status()
    result = response.json()
    if 'error' in result:
        raise ValueError(result['error'])
    page = result['parse']
    return {'title': page['title'], 'pageid': page['pageid'],
            'revision': page['revid'], 'html': page['text']['*']}


def clean(html):
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.select_one('.mw-parser-output')
    if body is None:
        raise ValueError('Missing article body')
    # Some wiki links are adjacent with no separating whitespace in source markup.
    for anchor in list(body.select('a')):
        following = anchor.next_sibling
        if getattr(following, 'name', None) == 'a':
            if anchor.get_text()[-1:].isalnum() and following.get_text()[:1].isalnum():
                anchor.insert_after(' ')
    for node in body.select('aside, table, nav, figure, script, style, '
                            '.toc, .thumb, .reference, .mw-editsection, .metadata, '
                            '.messagebox, .noprint, .references-small'):
        node.decompose()
    blocks, links, inline = [], [], []
    def flush():
        value = re.sub(r'\s+', ' ', ''.join(inline)).strip()
        if value:
            blocks.append(value)
        inline.clear()
    for node in body.contents:
        if isinstance(node, Comment):
            continue
        if isinstance(node, NavigableString):
            inline.append(str(node))
            continue
        if node.name in ('a', 'b', 'i', 'span', 'em', 'strong', 'sup', 'sub'):
            inline.append(node.get_text())
            anchors = [node] if node.name == 'a' else node.select('a[href^="/wiki/"]')
            links.extend((a.get_text(), a.get('href', '')) for a in anchors)
            continue
        flush()
        if node.name in ('h2', 'h3', 'h4'):
            label = node.get_text(' ', strip=True)
            if label.casefold() in STOP:
                break
            blocks.append('#' * int(node.name[1]) + ' ' + label)
        elif node.name in ('p', 'ul', 'ol', 'blockquote') or 'quote' in node.get('class', []):
            # Empty separator preserves punctuation and possessives around inline tags.
            if 'quote' in node.get('class', []):
                for br in node.select('br'):
                    br.replace_with('\n')
                value = '> ' + re.sub(r'\s+', ' ', node.get_text()).strip()
            else:
                value = re.sub(r'\s+', ' ', node.get_text()).strip()
            if value:
                blocks.append(value)
                links.extend((a.get_text(), a.get('href', '')) for a in node.select('a[href^="/wiki/"]'))
    flush()
    return '\n\n'.join(blocks) + '\n', links


def word_count(text):
    return len(re.findall(r"\b[\w]+(?:[-'’][\w]+)*\b", text))


def main():
    sources = read_json(BASE / 'sources.json')
    if len(sources) != 30 or len({s['pageid'] for s in sources}) != 30:
        raise ValueError('Expected 30 distinct sources')
    cache = load_cache()
    prepared = BASE / 'prepared'
    prepared.mkdir(exist_ok=True)
    for source in sources:
        key = source['title']
        page = cache.get(key)
        if page is None or page['revision'] != source['revision']:
            page = download(key, source['revision'])
            cache[key] = page
            save_cache(cache)
        if page['pageid'] != source['pageid']:
            raise ValueError('Source identity mismatch: ' + key)
        text, _ = clean(page['html'])
        if not 150 <= word_count(text) <= 400:
            raise ValueError('Unexpected article length: ' + key)
        title = key.removesuffix('/Legends')
        (prepared / source['file']).write_text('# ' + title + '\n\n' + text, encoding='utf-8')
    print('Prepared: 30 documents. Dictionary and final corpus untouched.')


if __name__ == '__main__':
    main()
