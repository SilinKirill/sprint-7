"""Download selected public wiki pages once; keep source HTML in one gzip cache."""
import argparse
import gzip
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent
API = "https://starwars.fandom.com/ru/api.php"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    sources = json.loads((BASE / "sources.json").read_text(encoding="utf-8"))
    cache_path = BASE / "sources_cache.json.gz"
    cache = json.loads(gzip.decompress(cache_path.read_bytes())) if cache_path.exists() else {}
    session = requests.Session()
    session.headers["User-Agent"] = "Sprint7Study/2.0 (educational project)"
    for source in sources[:args.limit]:
        sid = source["id"]
        if sid in cache:
            continue
        params = {"action": "parse", "format": "json", "prop": "text|revid"}
        params.update({"oldid": source["revision"]} if "revision" in source else {"pageid": source["pageid"]})
        response = session.get(API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise SystemExit(payload["error"])
        article = payload["parse"]
        soup = BeautifulSoup(article["text"]["*"], "html.parser")
        body = soup.select_one(".mw-parser-output")
        if body is None:
            raise SystemExit("Missing article body: " + sid)
        for node in body.select("aside, table, nav, figure, script, style, blockquote, .toc, .thumb, .reference, .mw-editsection, .metadata, .quote"):
            node.decompose()
        cache[sid] = {**source, "revision": article["revid"], "html": article["text"]["*"],
                      "text": body.get_text(" ", strip=True)}
        cache_path.write_bytes(gzip.compress(json.dumps(cache, ensure_ascii=False).encode("utf-8")))
        print("Downloaded " + sid, flush=True)
        time.sleep(0.5)
    for source in sources:
        if source["id"] in cache:
            source["revision"] = cache[source["id"]]["revision"]
            source["url"] = "https://starwars.fandom.com/ru/wiki/" + quote(source["title"].replace(" ", "_"))
    (BASE / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
