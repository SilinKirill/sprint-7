"""Require every identifying condition to match the retrieved evidence."""
from shared7 import rag
import requests

GROUNDING = '''
Before answering, check every identifying condition in the question against the
evidence for the SAME entity. Model designation, manufacturer, group, place,
date and negation are constraints, not optional hints. A similar entity that
matches only some conditions is not an answer. Never replace the requested model
or attribute with a different one. If no entity is supported for all requested
conditions, set answerable=false with empty evidence and explanation. Do not
infer missing conditions from shared manufacturers or organizations.
Answer the question directly in the answer field, rather than just repeating an
excerpt. For a who/which question, include the entity's name in answer, not only
in explanation. A passage title identifies its subject, but does not prove other
properties. Explain the supporting facts in 2-3 short items. Select only the
1-3 quote IDs necessary to support that concise answer.
'''


class GroundedClient(rag.OllamaClient):
    def generate(self, messages):
        enriched = [dict(message) for message in messages]
        enriched[0]['content'] += GROUNDING
        schema = rag.generation_schema()
        # String-length constraints remain stripped for Ollama compatibility.
        # Bound arrays during generation instead of accepting excess citations.
        for field in ('explanation', 'evidence'):
            schema['properties'][field]['maxItems'] = 3
        payload = {'model': self.settings.llm_model, 'messages': enriched,
                   'stream': False, 'think': False, 'format': schema,
                   'keep_alive': '10m',
                   'options': {'temperature': 0, 'seed': 7, 'num_ctx': 8192, 'num_predict': 1024}}
        try:
            response = self.session.post(self.settings.ollama_url + '/api/chat', json=payload,
                                         timeout=(10, 240), allow_redirects=False)
            if response.status_code != 200:
                raise rag.ServiceError(f'Ollama generation failed: HTTP {response.status_code}')
            data = response.json()
            if not data.get('done') or data.get('done_reason') == 'length':
                raise rag.OutputError('LLM response did not finish within the output limit')
            return data['message']['content']
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise rag.ServiceError('Local Ollama request failed') from exc
