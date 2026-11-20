"""Exercise ENCA framing and reply ownership with a scripted native worker."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine
from serve.service import Service

WORKER = r'''
import sys
from pathlib import Path

print("INFO batch_slots=1 vision_overlap=1", flush=True)
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    words = line.split()
    if words[0] != "ENCA":
        continue
    job = words[1]
    if words[2].endswith("exit"):
        break
    if words[2].endswith("refused"):
        print(f"ENCA {job} ERR independent vision is busy", flush=True)
        continue
    if words[2].endswith("timeout"):
        continue
    for i, out in enumerate(words[3::2]):
        assert Path(out).is_file()
        Path(out).write_bytes(b"embeddings")
        print(f"T {41 + i}", flush=True)
        print(f"BT 0 {91 + i}", flush=True)
        print(f"ENCA {job} OK {4 + i} 2 2 5", flush=True)
'''


class IndependentVisionProtocol(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        script = self.dir / "fake_engine"
        script.write_text(f"#!{sys.executable}\n" + WORKER)
        script.chmod(0o700)
        self.engine = ProcessEngine(str(script), [])
        self.addCleanup(self.engine.close)

    def test_image_replies_do_not_consume_solo_or_batch_lines(self):
        jobs = [(self.dir / f"img{i}", self.dir / f"out{i}") for i in range(2)]
        with self.engine.ctl:
            self.assertEqual(self.engine.encode_images(jobs, timeout=2), ["OK 4 2 2 5", "OK 5 2 2 5"])
        self.assertEqual([self.engine.lines.get(timeout=2).strip() for _ in jobs], ["T 41", "T 42"])
        self.assertEqual([self.engine.slot_q[0].get(timeout=2).strip() for _ in jobs], ["BT 0 91", "BT 0 92"])
        self.assertEqual([out.read_bytes() for _, out in jobs], [b"embeddings"] * 2)

    def test_refusal_and_timeout_release_the_reply_queue(self):
        with self.assertRaisesRegex(ValueError, "busy"):
            self.engine.encode_images([(self.dir / "refused", self.dir / "out1")], timeout=2)
        with self.assertRaisesRegex(ValueError, "timed out"):
            self.engine.encode_images([(self.dir / "timeout", self.dir / "out2")], timeout=0.05)
        self.assertFalse(self.engine.image_replies)
        self.assertTrue(self.engine.alive())

    def test_native_eof_releases_the_image_waiter(self):
        with self.assertRaisesRegex(EngineDied, "engine ended"):
            self.engine.encode_images([(self.dir / "exit", self.dir / "out")], timeout=2)
        self.assertFalse(self.engine.image_replies)

    def test_suspended_engine_declines_without_sending_job(self):
        self.engine.asleep = True
        try:
            with self.assertRaisesRegex(ValueError, "asleep"):
                self.engine.encode_images([(self.dir / "img", self.dir / "out")], timeout=2)
            self.assertFalse((self.dir / "out").exists())
        finally:
            self.engine.asleep = False

    def test_independent_image_preparation_does_not_take_text_fifo(self):
        svc = Service.__new__(Service)
        svc.images_lock = threading.Lock()
        svc.images_waiting = 0
        svc.embeddings = threading.local()
        svc.fifo = mock.MagicMock()
        svc.vision = SimpleNamespace(allows_overlap=True,
                                     read=lambda *_: [b"image"],
                                     encode=lambda _: ("one.sve", [4]))
        media = [{"type": "image"}]
        self.assertEqual(svc._encode_media(media, [object()], [], None), [4])
        svc.fifo.__enter__.assert_not_called()
        self.assertEqual(svc.embeddings.path, "one.sve")
