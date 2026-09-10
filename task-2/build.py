"""Build the V2 catalog locally. Only explicit entity names are replaced."""
import argparse
import json
import re
import secrets
from pathlib import Path

BASE = Path(__file__).resolve().parent


def replace_names(text, mapping):
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(name) for name in sorted(mapping, key=len, reverse=True)
    ) + r")(?!\w)", re.I)
    folded = {name.casefold(): value for name, value in mapping.items()}
    return pattern.sub(lambda match: folded[match.group().casefold()], text)


def inputs():
    sources = json.loads((BASE / "sources.json").read_text(encoding="utf-8"))
    if len(sources) < 30 or len({s["pageid"] for s in sources}) != len(sources):
        raise ValueError("Need at least 30 distinct source pages")
    documents = {}
    for source in sources:
        sid = source["id"]
        if not re.fullmatch(r"\d{2}", sid):
            raise ValueError("Invalid source ID")
        text = (BASE / "prepared" / (sid + ".md")).read_text(encoding="utf-8")
        title, body = text.split("\n", 1)
        if title != "# " + source["title"] or len(body.split()) < 20:
            raise ValueError("Missing heading or description: " + sid)
        if re.search(r"https?://|<[^>]+>|star\s*wars|зв[её]здн\w*\s+войн\w*", text, re.I):
            raise ValueError("Source marker: " + sid)
        # The edited paragraph must use ordinary descriptive vocabulary. All source
        # entity names are confined to the heading, avoiding grammatical inflection.
        for entity in sources:
            if re.search(r"(?<!\w)" + re.escape(entity["title"]) + r"(?!\w)", body, re.I):
                raise ValueError("Entity name in prose requires review: " + sid)
        documents[sid] = text
    if len(set(documents.values())) != len(documents):
        raise ValueError("Duplicate documents")
    return sources, documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check prepared texts; do not generate names or files")
    args = parser.parse_args()
    try:
        sources, documents = inputs()
        if args.check:
            sizes = [len(text.split("\n", 1)[1].split()) for text in documents.values()]
            print(f"Ready: {len(documents)} documents; {sum(sizes)} words; {min(sizes)}-{max(sizes)} words per description.")
            print("No names generated; no output written.")
            return
        output, map_path = BASE / "knowledge_base", BASE / "terms_map.json"
        if output.exists() or map_path.exists():
            raise ValueError("V2 result already exists; nothing overwritten")
        mapping = {}
        used = set()
        for source in sources:
            code = secrets.token_hex(4).upper()
            while code in used:
                code = secrets.token_hex(4).upper()
            used.add(code)
            mapping[source["title"]] = source["replacement_type"] + " «КР-" + code + "»"
        result = {sid: replace_names(text, mapping) for sid, text in documents.items()}
        output.mkdir()
        for sid, text in result.items():
            (output / ("doc_" + sid + ".md")).write_text(text, encoding="utf-8")
        with map_path.open("x", encoding="utf-8") as stream:
            json.dump(mapping, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Created: {len(result)} documents; {len(mapping)} replacements.")
        print("Saved in task-2_v2/knowledge_base and task-2_v2/terms_map.json.")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
