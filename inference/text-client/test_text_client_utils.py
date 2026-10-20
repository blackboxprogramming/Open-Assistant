"""Text-client response-lifetime checks with transport and SSE boundaries isolated."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import text_client_utils as utils


class Response:
    def __init__(self, data=None, status=200, error=None, json_error=None):
        self.data = data
        self.status_code = status
        self.error = error
        self.json_error = json_error
        self.close_count = 0

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.data

    def close(self):
        self.close_count += 1


class HTTP:
    def __init__(self, stream, completed=None):
        self.stream = stream
        self.completed = completed
        self.token = "test-bearer-value"
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        if url.endswith("/configs/model_configs"):
            return Response([{"name": "local-model"}])
        if url.endswith("/auth/callback/debug"):
            return Response({"access_token": {"token_type": "bearer", "access_token": self.token}})
        if url.endswith("/events"):
            return self.stream
        return self.completed

    def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        return Response({"id": "chat" if url.endswith("/chats") else "message"})


def event(event_type, **values):
    return SimpleNamespace(event="message", data=json.dumps({"event_type": event_type, **values}))


class TextClientLifetimeTests(unittest.TestCase):
    def client(self, stream, events=(), completed=None):
        http = HTTP(stream, completed)
        client = utils.DebugClient("http://local.invalid", http_client=http)
        client.login("reader")
        client.create_chat()
        self.sse = patch.object(
            utils.sseclient, "SSEClient", return_value=SimpleNamespace(events=lambda: iter(events))
        )
        self.sse.start()
        self.addCleanup(self.sse.stop)
        return client, http

    def test_stream_completion_closes_response_and_preserves_tokens(self):
        response = Response()
        client, _ = self.client(
            response,
            [
                SimpleNamespace(event="ping", data="not JSON"),
                event("token", text="hello "),
                event("token", text="world"),
                event("message", content="hello world"),
                event("token", text="must not be emitted"),
            ],
        )
        self.assertEqual(list(client.send_message("hello", "local-model")), ["hello ", "world"])
        self.assertEqual(response.close_count, 1)

    def test_closing_generator_after_a_token_closes_response(self):
        response = Response()
        client, _ = self.client(response, [event("token", text="partial"), event("token", text="later")])
        stream = client.send_message("hello", "local-model")
        self.assertEqual(next(stream), "partial")
        stream.close()
        self.assertEqual(response.close_count, 1)

    def test_transport_error_event_closes_response(self):
        response = Response()
        client, _ = self.client(response, [SimpleNamespace(event="error", data="stream failed")])
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            list(client.send_message("hello", "local-model"))
        self.assertEqual(response.close_count, 1)

    def test_application_error_event_closes_response(self):
        response = Response()
        client, _ = self.client(response, [event("error", error="generation failed")])
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            list(client.send_message("hello", "local-model"))
        self.assertEqual(response.close_count, 1)

    def test_malformed_event_closes_response(self):
        response = Response()
        client, _ = self.client(response, [SimpleNamespace(event="message", data="{")])
        with self.assertRaises(RuntimeError):
            list(client.send_message("hello", "local-model"))
        self.assertEqual(response.close_count, 1)

    def test_rejected_stream_status_closes_response_before_parsing(self):
        response = Response(error=RuntimeError("HTTP rejected"))
        client, _ = self.client(response)
        with self.assertRaisesRegex(RuntimeError, "HTTP rejected"):
            list(client.send_message("hello", "local-model"))
        self.assertEqual(response.close_count, 1)

    def test_completed_message_fallback_closes_both_responses(self):
        response = Response(status=204)
        completed = Response({"content": "already completed"})
        client, _ = self.client(response, completed=completed)
        self.assertEqual(list(client.send_message("hello", "local-model")), ["already completed"])
        self.assertEqual((response.close_count, completed.close_count), (1, 1))

    def test_closing_fallback_generator_closes_both_responses(self):
        response = Response(status=204)
        completed = Response({"content": "already completed"})
        client, _ = self.client(response, completed=completed)
        stream = client.send_message("hello", "local-model")
        self.assertEqual(next(stream), "already completed")
        stream.close()
        self.assertEqual((response.close_count, completed.close_count), (1, 1))

    def test_failed_fallback_status_closes_both_responses(self):
        response = Response(status=204)
        completed = Response(error=RuntimeError("fallback rejected"))
        client, _ = self.client(response, completed=completed)
        with self.assertRaisesRegex(RuntimeError, "fallback rejected"):
            list(client.send_message("hello", "local-model"))
        self.assertEqual((response.close_count, completed.close_count), (1, 1))

    def test_failed_fallback_json_closes_both_responses(self):
        response = Response(status=204)
        completed = Response(json_error=ValueError("bad JSON"))
        client, _ = self.client(response, completed=completed)
        with self.assertRaisesRegex(ValueError, "bad JSON"):
            list(client.send_message("hello", "local-model"))
        self.assertEqual((response.close_count, completed.close_count), (1, 1))

    def test_stream_iterator_failure_closes_response(self):
        response = Response()

        def events():
            yield event("token", text="partial")
            raise OSError("connection interrupted")

        client, _ = self.client(response, events())
        with self.assertRaisesRegex(OSError, "connection interrupted"):
            list(client.send_message("hello", "local-model"))
        self.assertEqual(response.close_count, 1)

    def test_login_keeps_authorization_but_does_not_log_the_token(self):
        response = Response()
        client, http = self.client(response)
        with patch.object(utils, "logger", Mock()) as logger:
            client.login("reader")
        self.assertEqual(client.auth_headers, {"Authorization": f"Bearer {http.token}"})
        self.assertNotIn(http.token, repr(logger.mock_calls))


if __name__ == "__main__":
    unittest.main()
