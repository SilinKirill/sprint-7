"""Prompt wiring only; no corpus access or model calls."""
import unittest
from unittest.mock import Mock
from grounding7 import GroundedClient, GROUNDING, rag


class GroundingTests(unittest.TestCase):
    def test_general_constraint_prompt_preserves_original_messages(self):
        messages = [{'role': 'system', 'content': 'Original protection.'},
                    {'role': 'user', 'content': 'Synthetic question'}]
        client = GroundedClient(rag.Settings())
        client.session = Mock()
        response = client.session.post.return_value
        response.status_code = 200
        response.json.return_value = {'done': True, 'message': {'content': 'synthetic'}}
        result = client.generate(messages)
        self.assertEqual(result, 'synthetic')
        payload = client.session.post.call_args.kwargs['json']
        sent = payload['messages']
        for field in ('explanation', 'evidence'):
            self.assertEqual(payload['format']['properties'][field]['maxItems'], 3)
        self.assertEqual(sent[0]['content'], 'Original protection.' + GROUNDING)
        self.assertEqual(sent[1], messages[1])
        self.assertEqual(messages[0]['content'], 'Original protection.')


if __name__ == '__main__':
    unittest.main()
