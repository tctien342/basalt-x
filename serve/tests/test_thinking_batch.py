import http.client
import json
import threading
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import mini_batch
from serve.config import build_parser, model_settings
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate, OutputParser
from serve.models.qwen4exp import Qwen4Exp
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer


def batch(*calls):
    return json.dumps({"calls": calls})


def calc(name, expr):
    return {"id": name, "tool": "calculate", "expr": expr}


class BatchTests(unittest.TestCase):
    def test_exact_order_and_errors(self):
        rows = mini_batch.execute(batch(calc("a", "comb(60, 30)"), calc("b", "a/3"),
                                        calc("bad", "1/0"), calc("miss", "bad+1"),
                                        calc("a", "0"), calc("c", "b*3")), "", 1)
        self.assertEqual(rows[0]["value"], "118264581564861424")
        self.assertEqual(rows[1]["value"], "118264581564861424/3")
        self.assertIn("error", rows[2])
        self.assertIn("missing earlier", rows[3]["error"])
        self.assertIn("duplicate", rows[4]["error"])
        self.assertEqual(rows[5]["value"], rows[0]["value"])
        self.assertIn("error", mini_batch.execute(batch(calc("x", "a")), "", 2)[0])

    def test_limits_and_tools(self):
        self.assertEqual(mini_batch.calculate("+".join(["1"] * 48), {}), 48)
        for expr in ("2**99999", "1e1000000", "(1).__class__", "open('x')", "comb(999999,4)"):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                mini_batch.calculate(expr, {})
        self.assertEqual(mini_batch.calculate("0.1+0.2", {}), mini_batch.Fraction(3, 10))
        rows = mini_batch.execute(batch(
            {"id": "hits", "tool": "regex", "operation": "count", "pattern": r"ORD-\d+"},
            {"id": "time", "tool": "datetime", "operation": "convert", "timezone": "Asia/Ho_Chi_Minh",
             "value": "2026-11-12T09:30:00+00:00"}), "ORD-17 ORD-35", 1)
        self.assertEqual(rows[0]["value"], 2)
        self.assertEqual(rows[1]["value"], "2026-11-12T16:30:00+07:00")
        self.assertNotIn("</think>", mini_batch.packed([{"value": "</think>"}]))

    def test_datetime_difference_units_and_iso_forms(self):
        pairs = [
            ("2026-11-03", "2026-11-12", "UTC", 9),
            ("2026-11-03T16:53:55Z", "2026-11-03T16:54:00Z", "UTC", 5),
            ("2026-11-03T16:53:55+00:00", "2026-11-03T23:53:55+07:00", "UTC", 0),
            ("2026-11-03 16:53:55", "2026-11-03T09:53:54.75Z", "Asia/Ho_Chi_Minh", -0.25),
            ("2026-03-08T01:30:00", "2026-03-08T03:30:00", "America/New_York", 3600),
        ]
        for start, end, timezone, expected in pairs:
            with self.subTest(start=start, end=end):
                rows = mini_batch.execute(batch({"id": "diff", "tool": "datetime",
                    "operation": "difference", "start": start, "end": end, "timezone": timezone}), "", 1)
                self.assertEqual(rows[0].get("value"), expected, rows)

        now = mini_batch.date_result("now", timezone="Asia/Ho_Chi_Minh")
        self.assertEqual(mini_batch.date_result("difference", start=now, end=now), 0)
        self.assertEqual(mini_batch.date_result("convert", value="2026-11-12T10:16:47Z",
                                               timezone="Asia/Ho_Chi_Minh"), "2026-11-12T17:16:47+07:00")
        self.assertEqual(mini_batch.date_result("add_days", value="2026-11-03", days=9), "2026-11-12")
        with self.assertRaisesRegex(ValueError, "two dates or two timestamps"):
            mini_batch.date_result("difference", start="2026-11-03", end=now)

    def test_streaming_and_code_fence(self):
        parser = OutputParser()
        parser.mini_batch = True
        example = "```text\n<mini_batch>example</mini_batch>\n```\n"
        text = example + "<mini_batch>" + batch(calc("a", "2+3")) + "</mini_batch>"
        events = [event for char in text for event in parser.feed(char)]
        self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), example)
        self.assertEqual(sum(e.kind == "mini_batch" for e in events), 1)
        self.assertEqual(parser.state, "reasoning")
        self.assertEqual(parser.feed("</think>\nOK")[0].text, "OK")

    def test_unterminated_batch_never_leaks(self):
        parser = OutputParser()
        parser.mini_batch = True
        self.assertEqual(parser.feed("<mini_batch>secret"), [])
        self.assertEqual(parser.finish(), [])

    def service(self, scripts, enabled=True):
        class ThinkTokenizer(ByteTokenizer):
            SPECIALS = [*ByteTokenizer.SPECIALS, "<think>", "</think>"]

        tok = ThinkTokenizer()
        engine = MockEngine(tok, scripts, max_context=8192)
        template = ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja")
        return Service(engine, tok, template, dialect=Qwen4Exp(), mini_tools_helper=enabled), engine

    def test_continuation_extends_exact_ids(self):
        first = "Checking.\n<mini_batch>" + batch(calc("a", "2+3"), calc("b", "a*2")) + "</mini_batch>"
        for effort in ("xhigh", "max"):
            with self.subTest(effort=effort):
                svc, engine = self.service([first, "</think>\n\n10"])
                ids, thinking, cap = svc.prepare([{"role": "user", "content": "compute"}], None,
                                                  {"reasoning_effort": effort}, 1024)
                events = list(svc.run(ids, thinking, None, cap, {"_thinking_max": effort=="max"}, threading.Event()))
                seen = [x for kind, x in events if kind == "event"]
                self.assertEqual("".join(x.text for x in seen if x.kind == "content"), "10")
                self.assertIn("🧮 Tool batch** · 2/2 succeeded", "".join(x.text for x in seen if x.kind == "reasoning"))
                self.assertNotIn("<mini_batch>", "".join(x.text for x in seen))
                self.assertEqual(len(engine.requests), 2)
                a, b = [p for p, _ in engine.requests]
                expected = a + svc.tok.encode(first, parse_special=True)
                self.assertEqual(b[:len(expected)], expected)
                self.assertIn('"value":"10"', svc.tok.decode(b[len(expected):]))
                self.assertTrue(all(("_think_close" in params) == (effort=="max") for _, params in engine.requests))

    def test_private_frame_after_client_call_does_not_leak_or_run(self):
        frame = "<mini_batch>" + batch(calc("forbidden", "777")) + "</mini_batch>"
        client_call = "<tool_call>\n<function=read_file>\n</function>\n</tool_call>"
        text = "</think>\nVisible answer\n" + client_call + "\n" + frame + "\nOK"
        for split in range(len(text) + 1):
            parser = OutputParser()
            parser.mini_batch = True
            events = parser.feed(text[:split]) + parser.feed(text[split:]) + parser.finish()
            self.assertFalse(any(event.kind == "mini_batch" for event in events))
            self.assertEqual([event.call.name for event in events if event.kind == "tool_call"], ["read_file"])
            call_index = next(i for i, event in enumerate(events) if event.kind == "tool_call")
            self.assertEqual("".join(event.text for event in events[:call_index] if event.kind == "content").strip(),
                             "Visible answer")
            self.assertEqual("".join(event.text for event in events[call_index + 1:]
                                     if event.kind == "content").strip(), "OK")

    def test_responses_stream_keeps_only_client_tool_events(self):
        for effort in ("xhigh", "max"):
            for before, lead in (("Checking.\n", ""), ("</think>\n\n", ""), ("</think>\n\n", "Redoing it properly:\n\n")):
                with self.subTest(effort=effort, before=before, lead=lead):
                    self.check_responses_batch(effort, before, lead)

    def check_responses_batch(self, effort, before, lead):
        frame = ("<tool_call>\n<function=engine_mini_batch>\n<parameter=calls>\n"
                 + json.dumps([calc("verified", "3**2*12312/12")])
                 + "\n</parameter>\n</function>\n</tool_call>")
        second = ("The calculator confirmed the value. I can now use it.\n</think>\n\n9234\n<tool_call>\n<function=read_file>\n"
                  "<parameter=path>\nfixture.txt\n</parameter>\n</function>\n</tool_call>\n"
                  "<tool_call>\n<function=client_custom>\n<parameter=input>\nopaque input\n"
                  "</parameter>\n</function>\n</tool_call>")
        svc, engine = self.service([before + lead + frame, second])
        httpd = serve(svc, port=0)
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=15)
        try:
            req = {"model": svc.model, "input": "Check and then call the client tools.",
                   "reasoning": {"effort": effort}, "stream": True, "max_output_tokens": 2048,
                   "tools": [
                       {"type": "function", "name": "read_file", "parameters":
                        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
                       {"type": "custom", "name": "client_custom", "description": "Accept opaque input."},
                   ]}
            conn.request("POST", "/v1/responses", json.dumps(req), {"Content-Type": "application/json"})
            response = conn.getresponse()
            payload = response.read().decode()
            self.assertEqual(response.status, 200, payload)
            events = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: {")]
            done = next(e["response"] for e in events if e["type"] == "response.completed")
            calls = [item for item in done["output"] if item["type"] in ("function_call", "custom_tool_call")]
            self.assertEqual([(item["type"], item["name"]) for item in calls],
                             [("function_call", "read_file"), ("custom_tool_call", "client_custom")])
            self.assertEqual(json.loads(calls[0]["arguments"]), {"path": "fixture.txt"})
            self.assertEqual(calls[1]["input"], "opaque input")
            self.assertEqual(len(engine.requests), 2)
            self.assertNotIn("<mini_batch>", payload)
            self.assertNotIn("<mini_result>", payload)
            self.assertNotIn(mini_batch.TOOL_NAME, payload)
            self.assertIn(mini_batch.TOOL_NAME, svc.tok.decode(engine.requests[0][0]))
            grammar_names = [tool["function"]["name"] for tool in engine.requests[0][1]["_grammar"]["tools"]]
            self.assertEqual(grammar_names, ["read_file", "client_custom", mini_batch.TOOL_NAME])
            reasoning = "".join(item["text"] for part in done["output"] if part["type"] == "reasoning"
                                for item in part["content"])
            self.assertIn("🧮 Tool batch** · 1/1 succeeded", reasoning)
            self.assertIn("9234", reasoning)
            self.assertIn("The calculator confirmed the value. I can now use it.", reasoning)
            self.assertTrue(any(event["type"] == "response.reasoning_text.delta"
                                for event in events[1:]))
            self.assertEqual("_think_close" in engine.requests[1][1], effort == "max")
            answer = "".join(item["text"] for part in done["output"] if part["type"] == "message"
                             for item in part["content"] if item["type"] == "output_text")
            self.assertEqual(answer.strip(), lead + "9234")
            first_ids = engine.requests[0][0] + svc.tok.encode(before + lead + frame, parse_special=True)
            next_ids = engine.requests[1][0]
            self.assertEqual(next_ids[:len(first_ids)], first_ids)
            suffix = svc.tok.decode(next_ids[len(first_ids):])
            self.assertIn("<|im_end|>\n<|im_start|>user\n<tool_response>", suffix)
            self.assertIn('"value":"9234"', suffix)
            self.assertTrue(suffix.endswith("</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"))
        finally:
            conn.close()
            httpd.shutdown()
            httpd.server_close()

    def test_native_batch_can_think_and_call_a_second_batch(self):
        def frame(name, expr):
            return ("<tool_call>\n<function=engine_mini_batch>\n<parameter=calls>\n"
                    + json.dumps([calc(name, expr)]) + "\n</parameter>\n</function>\n</tool_call>")

        svc, engine = self.service(["</think>\n\n" + frame("a", "6*7"),
                                    "Now double the result.\n" + frame("b", "42*2"),
                                    "Both results verified.\n</think>\n\n84"])
        ids, thinking, cap = svc.prepare([{"role": "user", "content": "Compute."}], None,
                                         {"reasoning_effort": "max"}, 1024)
        records = list(svc.run(ids, thinking, None, cap, {"_thinking_max": True}, threading.Event()))
        events = [event for kind, event in records if kind == "event"]
        reason = "".join(event.text for event in events if event.kind == "reasoning")
        self.assertEqual(reason.count("1/1 succeeded"), 2)
        self.assertIn("Now double the result.", reason)
        self.assertIn("Both results verified.", reason)
        self.assertEqual("".join(event.text for event in events if event.kind == "content"), "84")
        self.assertEqual(len(engine.requests), 3)
        self.assertTrue(all("_think_close" in sampling for _, sampling in engine.requests))

    def test_native_private_frames_across_stream_splits(self):
        frame = ("<tool_call>\n<function=engine_mini_batch>\n<parameter=calls>\n"
                 + json.dumps([calc("a", "6*7")]) + "\n</parameter>\n</function>\n</tool_call>")
        for prefix, suffix, execute in (
            ("Thinking.\n", frame, True),
            ("</think>\nLead.\n", frame, True),
            ("</think>\n```text\n", frame + "\n```", False),
            ("</think>\n", frame.removesuffix("</tool_call>"), False),
        ):
            text = prefix + suffix
            for split in range(len(text) + 1):
                parser = OutputParser(thinking=True, stream_tools=True)
                parser.mini_batch = True
                events = parser.feed(text[:split]) + parser.feed(text[split:]) + parser.finish()
                self.assertEqual(sum(e.kind == "mini_native_batch" for e in events), int(execute), split)
                self.assertFalse(any(e.kind.startswith("tool_") for e in events), split)
                visible = "".join(e.text for e in events if e.kind in ("reasoning", "content"))
                self.assertEqual("engine_mini_batch" in visible, prefix.endswith("```text\n"), split)

    def test_schema_policy_preserves_explicit_client_choices(self):
        svc, _ = self.service(["OK"])
        for choice in ("none", "required", {"type": "function", "name": "bash"}):
            req, kw = {"tool_choice": choice}, {}
            svc.configure_mini_tools(req, [], kw)
            self.assertFalse(req["_mini_tools_enabled"])
            self.assertNotIn(mini_batch.TOOL_NAME, svc.render_prompt([{"role": "user", "content": "Compute."}], None, kw))
        with self.assertRaisesRegex(ValueError, "reserved"):
            svc.configure_mini_tools({}, [mini_batch.TOOL], {})

    def test_flag_is_off_by_default_and_config_can_enable_it(self):
        self.assertFalse(build_parser().parse_args([]).mini_tools_helper)
        self.assertTrue(build_parser().parse_args(["--mini-tools-helper"]).mini_tools_helper)
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "basalt.json"
            cfg.write_text(json.dumps({"version": 2, "models": [{"model": "mock.basalt"}],
                                       "server": {"mini-tools-helper": True}}))
            opts = model_settings(build_parser(), ["--config", str(cfg)])[0][0]
            self.assertTrue(opts.mini_tools_helper)

    def test_disabled_service_leaves_prompt_and_output_parser_alone(self):
        frame = "<mini_batch>" + batch(calc("private", "6*7")) + "</mini_batch>"
        svc, engine = self.service([frame + "</think>\n\nOK"], enabled=False)
        ids, thinking, cap = svc.prepare([{"role": "user", "content": "compute"}], None,
                                          {"reasoning_effort": "xhigh"}, 1024)
        self.assertNotIn(mini_batch.INSTRUCTIONS, svc.tok.decode(ids))
        events = list(svc.run(ids, thinking, None, cap, {}, threading.Event()))
        self.assertEqual(len(engine.requests), 1)
        self.assertFalse(any(kind == "event" and "🧮 Tool batch" in event.text for kind, event in events))

    def test_prompt_instruction_is_stable_and_off_is_clean(self):
        svc, _ = self.service(["OK"])
        one = [{"role":"system","content":"Fixed prefix."},{"role":"user","content":"First."}]
        two = one + [{"role":"assistant","content":"One."},{"role":"user","content":"Second."}]
        a = svc.render_prompt(one, None, {"reasoning_effort":"xhigh"})
        b = svc.render_prompt(two, None, {"reasoning_effort":"xhigh"})
        self.assertEqual(a.split("First.")[0], b.split("First.")[0])
        for effort in ("xhigh", "max"):
            prompt = svc.render_prompt(one, None, {"reasoning_effort": effort})
            self.assertEqual(prompt.count(mini_batch.INSTRUCTIONS), 1)
            self.assertIn("unverified until a successful calculator result", prompt)
            self.assertIn("including incidental totals, estimates and unit conversions", prompt)
        self.assertNotIn(mini_batch.TOOL_NAME, svc.render_prompt(one, None, {"enable_thinking":False}))


class ProcessBoundary(unittest.TestCase):
    def test_text_stop_drain_and_resume_accounting(self):
        root = Path(__file__).resolve().parents[2]
        frame = "<mini_batch>" + batch(calc("a", "6*7")) + "</mini_batch>"
        source = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from serve.util.text import ByteTokenizer
tok = ByteTokenizer()
rd = sys.stdin.buffer
def say(text):
    print(text, flush=True)
def record(value):
    with open(sys.argv[2], "a") as log:
        print(json.dumps(value), file=log)
say("INFO engine=fake text_prompt=1")
say("READY 8192 stop")
last = []
while True:
    line = rd.readline().decode()
    if not line or line.startswith("QUIT"):
        break
    f = line.split()
    if f[0] == "GENT":
        size = int(next(word for word in f[2:] if "=" not in word))
        ids = tok.encode(rd.read(size).decode(), parse_special=True)
        say(f"PT {len(ids)} {f[1]}")
    else:
        assert f[0] == "GEN", f
        ids = [int(x) for x in f[-1].split(",")]
    record({"command": f[0], "ids": ids})
    reused = len(last) if ids[:len(last)] == last else 0
    output = tok.encode("</think>\n\n42" if last else sys.argv[3], parse_special=True)
    for token in output:
        say(f"T {token}")
    if not last:
        assert rd.readline().decode().strip() == "STOP"
        record({"command": "STOP"})
        last = ids + output
        finish = "cancel"
    else:
        say("T 257")
        finish = "stop"
    say(f"DONE {len(output)} {len(ids)} 1.0 2.0 {finish} 0 0 {reused} 0 0 0 0 0 {len(ids)-reused}")
"""
        with tempfile.TemporaryDirectory() as directory:
            script, seen = Path(directory) / "fake.py", Path(directory) / "seen.jsonl"
            script.write_text(source)
            popen = subprocess.Popen
            with mock.patch.object(subprocess, "Popen", lambda cmd, **kw: popen(
                    [sys.executable, str(script), str(root), str(seen), frame], **kw)):
                engine = ProcessEngine("basalt", [])
            try:
                tok = ByteTokenizer()
                svc = Service(engine, tok, ChatTemplate.from_file(root / "serve/chat_template.jinja"),
                              dialect=Qwen4Exp(), mini_tools_helper=True)
                ids, thinking, cap = svc.prepare([{"role": "user", "content": "Verify arithmetic."}], None,
                                                  {"reasoning_effort": "xhigh"}, 1024)
                events = list(svc.run(ids, thinking, None, cap, {}, threading.Event()))
                self.assertEqual("".join(x.text for kind, x in events if kind == "event" and x.kind == "content"), "42")
                records = [json.loads(line) for line in seen.read_text().splitlines()]
                self.assertEqual([row["command"] for row in records], ["GENT", "STOP", "GEN"])
                expected = records[0]["ids"] + tok.encode(frame, parse_special=True)
                self.assertEqual(records[2]["ids"][:len(expected)], expected)
                timings = events[-1][1]["timings"]
                self.assertEqual(timings["cache_n"], 0)
                self.assertEqual(timings["mini_segment_cache"][1]["reused"], len(expected))
                self.assertEqual(timings["mini_segment_cache"][1]["fresh"], len(records[2]["ids"]) - len(expected))
                self.assertIn('"value":"42"', tok.decode(records[2]["ids"][len(expected):]))
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
