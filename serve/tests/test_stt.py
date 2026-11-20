"""Regression tests for speech input normalization and CPU queue admission."""
import threading
import unittest

from serve.api.responses import input_messages
from serve.engine.stt import SpeechBusy, SpeechWorker
from serve.frontend import openai_to_messages


class SpeechInputs(unittest.TestCase):
    def test_chat_and_responses_preserve_audio_order(self):
        content = [
            {"type": "input_audio", "input_audio": {"format": "wav", "data": "AAAA"}},
            {"type": "audio_url", "audio_url": {"url": "https://example.test/audio.wav"}},
        ]
        messages = [{"role": "user", "content": content}]
        wanted = [
            {"type": "audio", "source": "data:audio/wav;base64,AAAA"},
            {"type": "audio", "source": "https://example.test/audio.wav"},
        ]
        self.assertEqual(openai_to_messages({"messages": messages})[0][0]["content"], wanted)
        self.assertEqual(input_messages({"input": messages})[0]["content"], wanted)


class SpeechAdmission(unittest.TestCase):
    def test_fifo_saturation_and_readmission(self):
        worker = SpeechWorker("unused", threads=1, waiting=1)
        started, release = threading.Event(), threading.Event()
        order = []

        def first():
            order.append("first")
            started.set()
            self.assertTrue(release.wait(5))

        try:
            a = worker.submit(first)
            self.assertTrue(started.wait(5))
            b = worker.submit(lambda: order.append("second"))
            with self.assertRaises(SpeechBusy):
                worker.submit(lambda: order.append("excess"))
            release.set()
            self.assertEqual(a.result(5), "")
            self.assertEqual(b.result(5), "")
            self.assertEqual(worker.submit(lambda: order.append("third")).result(5), "")
            self.assertEqual(order, ["first", "second", "third"])
        finally:
            release.set()
            worker.close()

    def test_video_budget_recovers_after_release(self):
        worker = SpeechWorker("unused")
        try:
            worker.retain_video(8, 10)
            with self.assertRaises(SpeechBusy):
                worker.retain_video(3, 10)
            worker.release_video(8)
            worker.retain_video(10, 10)
            worker.release_video(10)
        finally:
            worker.close()
