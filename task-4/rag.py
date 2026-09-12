"""Local RAG for CLI and Telegram. Imports never read the knowledge base."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prompts import messages_for, quote_options

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RETRIEVAL_CANDIDATES = 12
UNKNOWN = "I don't know: the knowledge base does not contain enough information."


class ServiceError(RuntimeError):
    """An infrastructure failure must not count as a knowledge refusal."""


class OutputError(RuntimeError):
    """The LLM response is not usable as a sourced answer."""


class Evidence(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    passage_id: str = Field(min_length=1, max_length=32)
    quote: str = Field(min_length=1, max_length=1800)


class Answer(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    answerable: bool
    answer: str = Field(max_length=2000)
    explanation: list[str] = Field(max_length=3)
    evidence: list[Evidence] = Field(max_length=3)


class SelectedEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    passage_id: str
    quote_id: str


class GeneratedAnswer(Answer):
    evidence: list[SelectedEvidence] = Field(max_length=3)


def resolve_answer(content, passages):
    try:
        generated = GeneratedAnswer.model_validate_json(content)
    except ValidationError as exc:
        raise OutputError('LLM returned an invalid response structure') from exc
    data = generated.model_dump()
    data['evidence'] = []
    if generated.answerable:
        options = {(p['passage_id'], q['quote_id']): q['text']
                   for p in passages for q in quote_options(p['text'])}
        for evidence in generated.evidence:
            key = (evidence.passage_id, evidence.quote_id)
            if key not in options:
                raise OutputError('Answer selected an unknown passage or quote ID')
            data['evidence'].append({'passage_id': evidence.passage_id, 'quote': options[key]})
    return validate_answer(json.dumps(data, ensure_ascii=False), passages)


def generation_schema():
    # Ollama's grammar compiler can reject bounded strings. Keep the shape
    # during generation; Answer still enforces every limit after generation.
    def simplify(value):
        if isinstance(value, dict):
            return {key: simplify(item) for key, item in value.items()
                    if key not in ('minLength', 'maxLength', 'minItems', 'maxItems', 'title')}
        if isinstance(value, list):
            return [simplify(item) for item in value]
        return value
    return simplify(GeneratedAnswer.model_json_schema())


@dataclass(frozen=True)
class Settings:
    ollama_url: str = 'http://127.0.0.1:11434'
    llm_model: str = 'qwen3:8b'
    embedding_device: str = 'cpu'
    offline_embeddings: bool = True

    @classmethod
    def from_env(cls):
        load_dotenv(BASE / '.env', override=False)
        value = cls(ollama_url=os.getenv('OLLAMA_URL', cls.ollama_url).rstrip('/'),
                    llm_model=os.getenv('OLLAMA_MODEL', cls.llm_model),
                    embedding_device=os.getenv('EMBEDDING_DEVICE', 'cpu'),
                    offline_embeddings=os.getenv('EMBEDDING_OFFLINE', 'true').lower() == 'true')
        url = urlsplit(value.ollama_url)
        if url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1', '::1') or url.username or url.password or url.query or url.fragment or url.path:
            raise ValueError('OLLAMA_URL must be a local HTTP address, e.g. http://127.0.0.1:11434')
        if value.llm_model != 'qwen3:8b':
            raise ValueError('This task uses the local qwen3:8b model selected in task 1')
        if value.embedding_device not in ('cpu', 'cuda'):
            raise ValueError('EMBEDDING_DEVICE must be cpu or cuda')
        return value


def load_task3_index():
    # Reuse the verified loader and encoder without reading task-2 dictionaries.
    sys.path.insert(0, str(ROOT / 'task-3'))
    from search import load_index
    return load_index()


class Retriever:
    def __init__(self, settings):
        self.index, self.chunks, self.report = load_task3_index()
        from common import load_model, encode
        self.encode = encode
        self.model = load_model(settings.embedding_device,
                                revision=self.report['model_revision'],
                                offline=settings.offline_embeddings)

    def find(self, question):
        vector = self.encode(self.model, [question], query=True)
        scores, ids = self.index.search(vector, min(RETRIEVAL_CANDIDATES, self.index.ntotal))
        selected, seen = [], set()
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if chunk['source'] in seen:
                continue
            seen.add(chunk['source'])
            selected.append({**chunk, 'passage_id': f'C{len(selected)+1}',
                             'cosine_score': float(score)})
            if len(selected) == 3:
                break
        return selected

    def prompt_tokens(self, messages):
        # Same Qwen3 vocabulary; reserve extra space for chat-template markers.
        return 256 + sum(len(self.model.tokenizer.encode(m['content'], add_special_tokens=False))
                         + 32 for m in messages)


class OllamaClient:
    def __init__(self, settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.trust_env = False  # Local context never travels through a proxy.
        self.model_digest = None

    def check(self):
        try:
            response = self.session.get(self.settings.ollama_url + '/api/tags', timeout=10,
                                        allow_redirects=False)
            if response.status_code != 200:
                raise ServiceError(f'Ollama returned HTTP {response.status_code}')
            models = response.json()['models']
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise ServiceError('Ollama is unavailable. Start Ollama and run: ollama pull qwen3:8b') from exc
        matches = [m for m in models if m.get('name') == self.settings.llm_model]
        if not matches:
            raise ServiceError('Local model missing. Run: ollama pull qwen3:8b')
        self.model_digest = matches[0].get('digest')
        return matches[0]

    def generate(self, messages):
        payload = {'model': self.settings.llm_model, 'messages': messages,
                   'stream': False, 'think': False, 'format': generation_schema(),
                   'keep_alive': '10m',
                   'options': {'temperature': 0, 'seed': 7, 'num_ctx': 8192, 'num_predict': 1024}}
        try:
            response = self.session.post(self.settings.ollama_url + '/api/chat', json=payload,
                                         timeout=(10, 240), allow_redirects=False)
            if response.status_code != 200:
                raise ServiceError(f'Ollama generation failed: HTTP {response.status_code}')
            data = response.json()
            if not data.get('done') or data.get('done_reason') == 'length':
                raise OutputError('LLM response did not finish within the output limit')
            content = data['message']['content']
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise ServiceError('Ollama request failed; check that the local model is running') from exc
        # Do not return or log a model's hidden thinking field.
        return content


def normalize(text):
    return ' '.join(text.split())


def validate_answer(content, passages):
    try:
        answer = Answer.model_validate_json(content)
    except ValidationError as exc:
        raise OutputError('LLM returned an invalid response structure') from exc
    if not answer.answerable:
        return Answer(answerable=False, answer=UNKNOWN, explanation=[], evidence=[])
    if not answer.answer.strip() or not 2 <= len(answer.explanation) <= 3 or not answer.evidence:
        raise OutputError('Answer lacks text, a short explanation or evidence')
    if any(not step.strip() or len(step) > 800 for step in answer.explanation):
        raise OutputError('Invalid explanation')
    by_id = {p['passage_id']: p for p in passages}
    for evidence in answer.evidence:
        if evidence.passage_id not in by_id:
            raise OutputError('Answer cites a passage outside current retrieval')
        quote = normalize(evidence.quote)
        if len(quote) < 12 or quote not in normalize(by_id[evidence.passage_id]['text']):
            raise OutputError('Evidence is not an exact quotation from the cited passage')
    return answer


def load_examples(chunks, report):
    path = BASE / 'few_shot.json'
    if not path.exists():
        raise ValueError('Run task-4/prepare_examples.py first')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['index_sha256'] != report['artifact_sha256']['faiss.index']:
        raise ValueError('Few-shot examples belong to another index; regenerate them')
    examples = data['examples']
    if len(examples) != 2:
        raise ValueError('Expected two few-shot examples')
    by_id = {chunk['chunk_id']: chunk for chunk in chunks}
    for example in examples:
        p = example['passage']
        original = by_id.get(p['chunk_id'])
        if original is None or p['title'] != original['title'] or p['source'] != original['source'] or p['text'] not in original['text']:
            raise ValueError('Few-shot context does not match the saved index')
        validate_answer(json.dumps(example['response']), [p])
    return examples


class RAG:
    def __init__(self, settings, *, retriever=None, client=None, examples=None):
        self.settings = settings
        self.client = client or OllamaClient(settings)
        if client is None:
            self.client.check()
        self.retriever = retriever or Retriever(settings)
        self.examples = examples if examples is not None else load_examples(
            self.retriever.chunks, self.retriever.report)

    def ask(self, question, *, top_k=3):
        if type(top_k) is not int or not 1 <= top_k <= 3:
            raise ValueError('top_k must be 1, 2 or 3')
        question = question.strip()
        if not question or len(question) > 2000:
            raise ValueError('Question must contain 1–2000 characters')
        started = time.perf_counter()
        passages = self.retriever.find(question)[:top_k]
        if not passages:
            answer = Answer(answerable=False, answer=UNKNOWN, explanation=[], evidence=[])
        else:
            messages = messages_for(question, passages, self.examples)
            while self.retriever.prompt_tokens(messages) > 6800 and len(passages) > 1:
                passages.pop()
                messages = messages_for(question, passages, self.examples)
            if self.retriever.prompt_tokens(messages) > 6800:
                raise ValueError('Question and context exceed the model context budget')
            answer = resolve_answer(self.client.generate(messages), passages)
        return {'question': question, 'status': 'answered' if answer.answerable else 'unknown',
                'answer': answer.model_dump(), 'retrieved': passages,
                'seconds': round(time.perf_counter() - started, 3)}


def format_reply(result):
    answer = result['answer']
    if not answer['answerable']:
        return UNKNOWN
    lines = [answer['answer'], '', 'Explanation:']
    lines.extend(f'{i}. {step}' for i, step in enumerate(answer['explanation'], 1))
    by_id = {p['passage_id']: p for p in result['retrieved']}
    lines.extend(['', 'Sources:'])
    used = set()
    for evidence in answer['evidence']:
        p = by_id[evidence['passage_id']]
        if p['chunk_id'] not in used:
            lines.append(f"[{p['passage_id']}] {p['title']} — {p['source']}, "
                         f"lines {p['start_line']}–{p['end_line']}, chunk {p['chunk_id']}")
            used.add(p['chunk_id'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--query')
    group.add_argument('--repl', action='store_true')
    group.add_argument('--check', action='store_true', help='Check local Ollama without opening the index')
    parser.add_argument('--top-k', type=int, choices=(1, 2, 3), default=3,
                        help='Number of highest-ranked passages sent to the LLM (default: 3)')
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.check:
        model = OllamaClient(settings).check()
        print('Ollama OK. Local model:', model['name'])
        print('Quantization:', model.get('details', {}).get('quantization_level', 'unknown'))
        return
    engine = RAG(settings)
    if args.query is not None:
        print(format_reply(engine.ask(args.query, top_k=args.top_k)))
        return
    print('RAG ready. Enter /exit to quit. Each question is independent.')
    while True:
        try:
            question = input('\nQ: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == '/exit':
            break
        try:
            print(format_reply(engine.ask(question, top_k=args.top_k)))
        except (ValueError, ServiceError, OutputError) as exc:
            print('ERROR:', str(exc))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, ServiceError, OutputError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
