"""Reload complete snapshots between questions; use only their current examples."""
from pathlib import Path
from shared6 import BASE, rag, ProtectedRAG, load_model, encode, make_examples, filter_passages
from snapshot import read_snapshot


class SnapshotRetriever(rag.Retriever):
    def __init__(self, path, settings):
        self.path, self.settings = Path(path), settings
        self.model = None
        self.signature = None
        self.encode = encode
        self.reload()

    def reload(self):
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self.signature:
            return False
        index, chunks, report = read_snapshot(self.path)
        if self.model is not None and report['model_revision'] != self.report['model_revision']:
            raise ValueError('Encoder revision changed; restart the bot')
        self.index, self.chunks, self.report = index, chunks, report
        self.signature = signature
        return True

    def ensure_model(self):
        if self.model is None:
            self.model = load_model(self.settings.embedding_device, revision=self.report['model_revision'],
                                    offline=self.settings.offline_embeddings)

    def find(self, question):
        if self.index.ntotal == 0:
            return []
        self.ensure_model()
        return super().find(question)

    def prompt_tokens(self, messages):
        self.ensure_model()
        return super().prompt_tokens(messages)


def current_examples(retriever, excluded_sources=()):
    safe, _ = filter_passages(retriever.chunks, 'all')
    try:
        return make_examples(safe, retriever.report, set(excluded_sources))['examples']
    except ValueError:
        # An empty/tiny filtered corpus must still support a correct refusal.
        return []


class LiveRAG:
    def __init__(self, settings, *, snapshot_path=None, excluded_sources=(), client=None):
        self.settings = settings
        self.retriever = SnapshotRetriever(snapshot_path or BASE / 'index/snapshot.zip', settings)
        self.client = client or rag.OllamaClient(settings)
        if client is None:
            self.client.check()
        self.excluded_sources = set(excluded_sources)
        self._reset()

    def _reset(self):
        self.examples = current_examples(self.retriever, self.excluded_sources)
        self.engine = ProtectedRAG(self.settings, retriever=self.retriever, client=self.client, examples=self.examples)

    def ask(self, question):
        if self.retriever.reload():
            self._reset()
        result = self.engine.ask(question, mode='all')
        result['snapshot_sha256'] = self.retriever.report['snapshot_sha256']
        result['few_shot_count'] = len(self.examples)
        return result
