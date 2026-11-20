import threading
import unittest

from serve.config import build_parser, parse_settings
from serve.engine.errors import EngineError
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate
from serve.prompt import CTX_SLACK, PromptText
from serve.router import Router
from serve.service import Service
from serve.util.text import ByteTokenizer

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen3.8-flash-next"


class ContextProfiles(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        engine = MockEngine(self.tok, "Hello", max_context=1048576)
        template = ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")
        self.svc = Service(engine, self.tok, template, MODEL, atomic_lora=True, swift_lora=True,
                           context_profiles=True, fit_max_tokens=True)

    def test_exact_catalog_and_routing(self):
        router = Router([self.svc])
        wanted = {MODEL + adapter + suffix: cap for adapter in ("", "-uncen", "-swift")
                  for suffix, cap in (("", 262144), ("-512k", 524288))}
        self.assertEqual({row["id"]: row["meta"]["n_ctx"] for row in router.models()}, wanted)
        for model in wanted:
            self.assertIs(router.service_for(model), self.svc)
            self.assertEqual(self.svc.canonical_model(model), model)
        self.assertIsNone(router.service_for(MODEL + "-uncen-swift"))
        self.assertIsNone(router.service_for(MODEL + "-uncen-swift-512k"))

    def test_cap_is_independent_of_pool_for_text_and_ids(self):
        p = PromptText(self.tok, "", clip=True, context_limit=262144)
        p._ids = [7] * (262144 - CTX_SLACK - 1)
        self.assertEqual(p.resolve(80, 1048576)[1], 1)
        self.assertEqual(self.svc._fit(p._ids, {}, 80)[2], 1)
        p._ids.append(7)
        with self.assertRaises(ValueError):
            p.resolve(80, 1048576)
        with self.assertRaises(ValueError):
            self.svc._fit(p._ids, {}, 80)
        self.svc.request_ctx.model = MODEL + "-swift-512k"
        self.assertEqual(self.svc._fit(p._ids, {}, 80)[2], 80)

    def test_native_flag_and_capability_guard(self):
        args, _, _, words = parse_settings(build_parser(), ["--context-profiles"])
        self.assertTrue(args.shared_kv_pool)
        self.assertIn("shared_kv_pool=1", words)
        self.assertIn("rope_context_profiles=1", words)
        self.assertEqual(ProcessEngine.sampling_keys({"_rope_profile": 0}), " rope_profile=0")
        self.assertEqual(ProcessEngine.sampling_keys(self.svc.adapted_models[MODEL + "-swift-512k"]),
                         " swift_lora=1 rope_profile=1")
        engine = ProcessEngine.__new__(ProcessEngine)
        engine.alive = lambda: True
        engine.info = {"shared_kv_pages": 262144}
        engine.max_context = 1048576
        engine.batch = 1
        with self.assertRaisesRegex(EngineError, "does not support"):
            next(engine.generate([1], 8, {"_rope_profile": 1, "_context_limit": 524288}, threading.Event()))
