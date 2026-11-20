"""Config "parallel": N, requests decoding together in the engine's batch slots (--batch N).

A fake engine script speaking GEN / BGEN / BT / BDONE / BADM / BSTOP runs behind the real ProcessEngine and Service."""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.engine.process import EngineDied, ProcessEngine
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# A window gives a slot up to MULTI tokens in ONE line, `BT <slot> <id> [<id> ..]`.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257). A request with logprobs=K gets a
# P line ahead of each T line (BP <slot> ahead of a BT line): the pick at -0.25, then K alternatives.
FAKE_BATCH = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args        # the first window over two slots fails
ready = int(args[args.index("--ready") + 1]) if "--ready" in args else 4096
MULTI = int(args[args.index("--multi") + 1]) if "--multi" in args else 3
nopp = "--nopp" in args               # no PP lines while a prompt is read: a read beside the windows reports none
STEP = 0.02
s0 = "--slot0solo" in args           # slot 0 is the solo session: GEN copies the best slot's conversation over slot 0
CH = 4096 if s0 else 32
sstore = {}                           # slot -> the stream its storage holds (s0 only)
pms = "--pms" in args                 # a run that extends an earlier prompt re-reads next to nothing (0.4 ms), else 500 ms
seen = []                             # prompts (as id lists) of earlier runs
keep = "--reuse" in args              # a conversation cache: a prompt reuses the longest start some earlier stream had
held = []                             # streams (prompt + tokens read so far) of earlier requests
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.0.1" + (f" batch_slots={fit}" if fit >= 2 or (s0 and fit == 1) else "") +
      (" slot_cache=1" if "--slotcache" in args else "") + (" slot0_solo=1" if s0 else "") +
      (" solo_any_slot=1" if s0 and "--soloany" in args else ""), flush=True)
print(f"READY {ready} stop", flush=True)
LONG = list(b"LONGREPLY")
def reply(ids):            # the rest of the reply after what the prompt already ends with (a request continued)
    ids = [int(x) for x in ids]
    long_one = any(ids[i:i + len(LONG)] == LONG for i in range(len(ids)))
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    k = max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
lp_top = {}          # slot (None: solo) -> the request's logprobs=K
def lp_fields(t, k):
    return f"{t} -0.25" + "".join(f" {t if i == 0 else 32 + i}:{-0.25 - 2 * i}" for i in range(k))
stopped = set()      # BSTOPped slots: they end "cancel"
def window():        # one batch window: every active slot one token
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    for b in sorted(active):
        left, max_new, produced = active[b]
        got = []
        while True:
            t = left.pop(0) if left else 257
            produced += 1
            got.append(t)
            if t == 257 or produced >= max_new or len(got) >= MULTI:
                break
        if lp_top.get(b) is not None:
            for t2 in got:
                print(f"BP {b} " + lp_fields(t2, lp_top[b]), flush=True)
        print(f"BT {b} " + " ".join(map(str, got)), flush=True)
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            counts = f" {b + 3} {10 * (b + 1)}" if "--bdone-drafts" in args else ""   # accepted, offered
            if "--draft-counts" in args:      # as the engine: the slot's own windows (not the admission's token)
                counts = f" {produced - 1} {2 * (produced - 1)}"
            print(f"BDONE {b} {produced} {fin} 1.0{counts}", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith(("GEN ", "BGEN ", "SGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] in ("BGEN", "SGEN") else None
        max_new = int(f[2] if slot is not None else f[1])
        bslot = slot if f[0] == "BGEN" else None      # SGEN: a solo run on that slot's own state
        ids = f[-1].split(",")
        lp = next((int(x[9:]) for x in f if x.startswith("logprobs=")), None)
        lp_top[None] = lp
        if bslot is not None:
            lp_top[bslot] = lp
        if lp is not None and bytes(int(x) & 255 for x in ids).find(b"PIPELINED") >= 0:   # as a layer-split engine
            print("ERR sampling logprobs: pipelined batch groups are not supported", flush=True)
            continue
        if bytes(int(x) & 255 for x in ids).find(b"REFUSEME") >= 0:   # refused up front, then idle (no DONE)
            print("ERR refused: images are not enabled", flush=True)
            continue
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        toks = reply(ids)
        reused = min(max([lcp([int(x) for x in ids], h) for h in held], default=0), len(ids) - 1) if keep else 0
        if s0:
            src = [sstore.get(slot, [])] if slot is not None else list(sstore.values())
            reused = min(max([lcp([int(x) for x in ids], h) for h in src], default=0), len(ids) - 1)
        stop.clear()
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        for q in range(CH * (reused // CH + 1) if s0 else CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            if not nopp:
                print(f"PP {q} {len(ids)} 1 1", flush=True)
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None:
                        given = (ys, q)
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 0", flush=True)
            if bslot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = 1 if bslot is not None else max_new
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            if lp is not None:
                print("P " + lp_fields(t, lp), flush=True)
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        dc = f"{out} {2 * out}" if "--draft-counts" in args and bslot is None else "0 0"   # a solo run drafts
        mine_ids = [int(x) for x in ids]
        pm = (0.4 if any(mine_ids[:len(h)] == h for h in seen) else 500.0) if pms else 5.0
        seen.append(mine_ids)
        print(f"DONE {out} {len(ids)} {pm} {out * STEP * 1000:.1f} {fin} {dc} {reused}", flush=True)
        if s0:
            sstore[0 if slot is None else slot] = [int(x) for x in ids] + toks[:-1]
        if keep:
            mine = [int(x) for x in ids] + toks[:out]
            held[:] = [h for h in held if mine[:len(h)] != h] + [mine]
        if bslot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class PickSlot(unittest.TestCase):
    def engine(self, n):
        e = ProcessEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_used = [0.0] * n
        e.slot_owner = {}
        e.session_slot = {}
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6]), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9]), 0)
        self.assertEqual(e.pick_slot([7, 8]), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8]), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8]))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)                 # a held prompt is never the WHOLE prompt


    def test_shared_prefix_affinity_matches_adapter(self):
        e = self.engine(3)
        e.slot_adapters = [True, False, False]
        e.slot_held = [[1, 2, 3], [1, 2], []]
        e.claim_slot(0, "same-session")
        prompt = [1, 2, 3, 4]
        self.assertEqual(e.pick_slot(prompt, "same-session", adapter=False), 1)
        self.assertEqual(e.pick_slot(prompt, "same-session", adapter=True), 0)
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot(prompt, "same-session", adapter=False), 2)

    def test_shared_prefix_affinity_matches_full_adapter_combination(self):
        e = self.engine(4)
        e.slot_adapters = [0, 1, 2, 3]
        e.slot_held = [[7, 8, 9], [7, 8, 9], [7, 8, 9], [7, 8, 9]]
        e.claim_slot(3, "same-session")
        for adapter in range(4):
            self.assertEqual(e.pick_slot([7, 8, 9, 10], "same-session", adapter=adapter), adapter)

    def test_shared_cold_admission_preserves_the_larger_idle_prefix(self):
        e = self.engine(2)
        e.slot_adapters = [False, False]
        e.slot_held = [[1] * 285669, [2] * 22986]
        e.slot_used = [1.0, 2.0]
        prompt = [2] * 22497 + [3] * 837
        # The oldest slot's large prefix must survive a missed bridge hint
        # when the shorter conversation rewrote its previous turn.
        self.assertEqual(e.pick_slot(prompt, adapter=False), 1)
        self.assertEqual(e.pick_slot(e.slot_held[0] + [4], adapter=False), 0)
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot(prompt, adapter=False), 0)

    def test_shared_long_continuation_uses_the_slot_with_a_rewritten_tail(self):
        e = self.engine(2)
        e.slot_adapters = [False, False]
        common = [1] * 19419
        long_prefix = common + [2] * 336000
        e.slot_held = [common + [3] * 30000, long_prefix + [4] * 1747]
        prompt = long_prefix + [5] * 2538
        self.assertEqual(e.pick_slot(prompt, adapter=False), 1)
        e.slot_adapters[1] = True
        self.assertEqual(e.pick_slot(prompt, adapter=True), 1)
        self.assertEqual(e.pick_slot(prompt, adapter=False), 0)

    def test_shared_long_slot_zero_prefix_can_match_without_session_key(self):
        e = self.engine(2)
        e.info["cold_shared_recovery_min_tokens"] = 8192
        e.slot_adapters = [4, 2]
        e.slot_held = [[7] * 9000, [1, 2]]
        e.claim_slot(0, "yarn-owner")
        self.assertEqual(e.pick_slot([7] * 8900 + [3], adapter=4), 0)
        self.assertEqual(e.pick_slot([7] * 8900 + [3], adapter=2), 1)
        self.assertEqual(e.pick_slot([7] * 4000 + [3], adapter=4), 1)

    def test_shared_common_system_prefix_does_not_displace_a_long_conversation(self):
        e = self.engine(2)
        e.slot_adapters = [False, False]
        e.slot_held = [[1] * 19419 + [2] * 336000, []]
        self.assertEqual(e.pick_slot([1] * 19419 + [3], adapter=False), 1)

class ParallelService(unittest.TestCase):
    """The real ProcessEngine and Service over HTTP, the fake engine behind them."""

    def start(self, slots, fit=None, slot_cache=False, fail=False, ready=None, multi=None, drafts=False, counts=False,
              nopp=False, reuse=False, slot0_solo=False, solo_any=False, pms=False):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_engine.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--fail-window"] if fail else []
        extra += ["--bdone-drafts"] if drafts else []
        extra += ["--draft-counts"] if counts else []
        extra += ["--ready", str(ready)] if ready else []
        extra += ["--multi", str(multi)] if multi else []
        extra += ["--nopp"] if nopp else []
        extra += ["--reuse"] if reuse else []
        extra += ["--slot0solo"] if slot0_solo else []
        extra += ["--soloany"] if solo_any else []
        extra += ["--pms"] if pms else []
        with mock.patch.object(subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = ProcessEngine("basalt", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=64):
        body = {"model": "basalt", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # /metrics showed the running requests, and one was admitted while another decoded in a slot (the fake logs
        # how many slots were active at each admission)
        self.assertTrue(any(m["state"] == "generating" for m in seen))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """A long solo prompt read gives way at a chunk boundary to a short request, then continues and is answered."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.assertFalse(any(self.engine.slot_busy))

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_silent_admission_gives_way_without_progress_lines(self):
        """The engine reads a prompt beside the slots' windows and sends no progress line: a short request that
        arrives meanwhile still gets the engine at the next chunk boundary, and the long read finishes after it."""
        self.start(3, nopp=True)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        yields = [x for x in log if x.startswith("YIELD ")]
        self.assertEqual(len(yields), 1, log)
        # the long prompt was sent twice: the part read, then again from the slot that held it
        longs = [x.split()[2] for x in log if x.startswith("BGEN") and int(x.split()[2]) > 1000]
        self.assertEqual(len(longs), 2, log)
        self.assertEqual(longs[0], longs[1])
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_long_read_gives_way_a_bounded_number_of_times(self):
        """Short requests keep arriving: the read gives way at most YIELDS_MAX times, then runs to the end."""
        self.start(3, nopp=True)
        long_q = "long " * 700
        out, errors = {}, []

        def go(t, n=64):
            try:
                out[t] = self.chat(t, max_tokens=n)["choices"][0]["message"]["content"]
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)
        a = threading.Thread(target=go, args=(long_q,))
        a.start()
        shorts = []
        for i in range(8):
            time.sleep(0.25)
            t = threading.Thread(target=go, args=(f"short {i}",))
            t.start()
            shorts.append(t)
        for t in shorts + [a]:
            t.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(out[long_q], "ok, done.")
        self.assertEqual([out[f"short {i}"] for i in range(8)], ["ok, done."] * 8)
        yields = [x for x in self.log.read_text().splitlines() if x.startswith("YIELD ")]
        self.assertGreaterEqual(len(yields), 1)
        self.assertLessEqual(len(yields), ProcessEngine.YIELDS_MAX, yields)
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_request_waiting_for_a_slot_leaves_the_control_lines(self):
        """Every slot is taken: the request waits for one without holding the control lines (a read that gave way
        waits in its slot for them), and runs once a slot frees."""
        self.start(2)
        with self.engine.slot_cv:
            self.engine.slot_busy[:] = [True, True]
        res = {}
        th = threading.Thread(target=lambda: res.setdefault("r", self.chat("waits for a slot")))
        th.start()
        time.sleep(1.0)
        self.assertTrue(th.is_alive())
        self.assertFalse(self.engine.ctl.locked())
        self.assertFalse(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        with self.engine.slot_cv:
            self.engine.slot_busy[1] = False
            self.engine.slot_cv.notify_all()
        th.join(30)
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        with self.engine.slot_cv:
            self.engine.slot_busy[0] = False

    def test_incremental_pool_three_requests_continue_after_partial_grants(self):
        self._exercise_incremental_three(cancel_parked=False)

    def test_incremental_pool_cancels_parked_stream(self):
        self._exercise_incremental_three(cancel_parked=True)

    def _exercise_incremental_three(self, *, cancel_parked):
        with mock.patch(__name__ + ".FAKE_BATCH", FAKE_BATCH.replace('b"la " * 15', 'b"la " * 400')):
            self.start(3, multi=64)
        engine = self.engine
        engine.info.update(shared_kv_pages=704, shared_kv_page_cells=4, shared_kv_mixed_adapters=1)
        with engine.slot_cv:
            engine.slot_busy[0] = True
            engine.slot_kv_pages[0] = 352
            engine.slot_kv_full[0] = object()

        prompt = list(b"LONGREPLY xyz")
        want = list(b"ok, " + b"la " * 400 + b"done.") + [257]
        cancel = [threading.Event(), threading.Event()]
        out = [[], []]
        failures = []
        sent = []
        original = engine._send

        def record(line):
            sent.append(line)
            return original(line)

        def stream(i):
            try:
                out[i].extend(int(t) for t in engine.generate(
                    prompt, 1500, {"_atomic_lora": bool(i), "temperature": 0.5, "seed": i + 11},
                    cancel[i]) if t is not None)
            except Exception as err:
                failures.append(err)

        with mock.patch.dict(os.environ, {"BASALT_SHARED_KV_INCREMENTAL": "1"}), \
                mock.patch.object(engine, "_send", side_effect=record):
            threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(2)]
            try:
                for th in threads:
                    th.start()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    with engine.slot_cv:
                        if all(engine.slot_busy):
                            self.assertEqual([bool(owner) for owner in engine.slot_kv_full], [True, False, False])
                            self.assertLessEqual(sum(engine.slot_kv_pages), 704)
                            break
                    time.sleep(0.005)
                else:
                    self.fail(f"three owners never overlapped: {sent}")
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    with engine.slot_cv:
                        if engine.slot_busy == [True, False, False] and len(engine.admission) == 2:
                            break
                    time.sleep(0.005)
                else:
                    self.fail(f"both bounded streams did not park beside the full owner: {sent}")
                if cancel_parked:
                    cancel[1].set()
                    threads[1].join(3)
                    self.assertFalse(threads[1].is_alive())
                    with engine.slot_cv:
                        self.assertEqual(engine.slot_kv_pages, [352, 0, 0])
                        self.assertEqual(len(engine.admission), 1)
            finally:
                with engine.slot_cv:
                    engine.slot_kv_pages[0] = 0
                    engine.slot_busy[0] = False
                    engine.slot_cv.notify_all()
                if not cancel_parked:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        with engine.slot_cv:
                            if sum(engine.slot_busy) >= 2:
                                break
                        time.sleep(0.005)
                    else:
                        failures.append("the resumed streams did not overlap")
                for th in threads:
                    th.join(15)
        self.assertFalse(any(th.is_alive() for th in threads), sent)
        self.assertEqual(failures, [])
        self.assertEqual(out[0], want)
        if cancel_parked:
            self.assertGreater(len(out[1]), 0)
            self.assertLess(len(out[1]), len(want))
            self.assertEqual(out[1], want[:len(out[1])])
            self.assertEqual(sum(" seed=12 " in line for line in sent), 1)
        else:
            self.assertEqual(out[1], want)
        for seed in ((11,) if cancel_parked else (11, 12)):
            runs = [line.split() for line in sent if line.startswith("BGEN ") and f" seed={seed} " in line]
            self.assertGreaterEqual(len(runs), 2, sent)
            self.assertLess(int(runs[0][2]), 1500)
            self.assertTrue(runs[-1][-1].startswith(runs[0][-1] + ","))
        self.assertEqual(sum(engine.slot_kv_pages), 0)
        self.assertEqual(engine.queue_depth(), 0)

    def test_shared_indexer_retry_after_other_slot_is_reused(self):
        self._exercise_shared_backing_retry("indexer backing unavailable: the driver has no VRAM for the KV pools")

    def test_shared_draft_retry_after_other_slot_is_reused(self):
        self._exercise_shared_backing_retry("draft backing unavailable: mtp: allocating the slot's draft K/V failed")

    def _exercise_shared_backing_retry(self, detail):
        fake = FAKE_BATCH.replace("active = {}", "refused = False\nactive = {}")
        fake = fake.replace(
            '        ids = f[-1].split(",")',
            '        if bslot == 1 and not refused:\n'
            '            refused = True\n'
            f'            print({("ERR shared KV: " + detail)!r}, flush=True)\n'
            '            continue\n'
            '        ids = f[-1].split(",")')
        with mock.patch(__name__ + ".FAKE_BATCH", fake):
            self.start(2)
        engine = self.engine
        engine.info.update(shared_kv_pages=128, shared_kv_page_cells=4, shared_kv_mixed_adapters=1)
        with engine.slot_cv:
            engine.slot_busy[0] = True
            engine.slot_kv_pages[0] = 32
        outputs, errors = [], []
        cancel = threading.Event()
        first_send = threading.Event()
        send = engine._send

        def record(line):
            send(line)
            if line.startswith("BGEN 1 "):
                first_send.set()

        def generate():
            try:
                outputs.extend(t for t in engine.generate([1, 2, 3], 20, {}, cancel) if t is not None)
            except Exception as err:
                errors.append(err)

        with mock.patch.object(engine, "_send", side_effect=record):
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            try:
                self.assertTrue(first_send.wait(3))
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    with engine.slot_cv:
                        if engine.slot_kv_pages[1] == 0:
                            # Slot 0 finished and was reclaimed before the waiting stream could observe it idle.
                            engine.slot_used[0] += 1
                            engine.slot_cv.notify_all()
                            break
                    time.sleep(.01)
                else:
                    self.fail("native refusal did not release the provisional grant")
                thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(outputs, list(b"ok, done.") + [257])
            finally:
                cancel.set()
                with engine.slot_cv:
                    engine.slot_busy[0] = False
                    engine.slot_kv_pages[0] = 0
                    engine.slot_cv.notify_all()
                thread.join(3)
        self.assertEqual(engine.queue_depth(), 0)

    def test_shared_slot_zero_returns_to_solo_after_peer_drains(self):
        self._exercise_shared_solo_return("solo")

    def test_shared_solo_return_keeps_slot_when_a_waiter_arrives(self):
        self._exercise_shared_solo_return("waiter")

    def test_shared_solo_return_checks_cancellation_after_bdone(self):
        self._exercise_shared_solo_return("cancel")

    def test_shared_solo_return_checks_ledger_after_restart(self):
        self._exercise_shared_solo_return("restart")

    def _exercise_shared_solo_return(self, case):
        self.start(2, slot_cache=True, slot0_solo=True, solo_any=True)
        engine = self.engine
        engine.info.update(shared_kv_pages=1024, shared_kv_page_cells=4, shared_kv_mixed_adapters=1)
        engine.slot_busy[1] = True
        engine.slot_kv_pages[1] = 32
        send, take_control = engine._send, engine._take_control
        old_busy, old_pages = engine.slot_busy, engine.slot_kv_pages
        cancel = threading.Event()
        sent, charges, tickets, outputs = [], [], [], []
        stopped = False

        def record(line):
            nonlocal stopped
            if line.startswith(("BGEN 0 ", "BSTOP 0", "SGEN 0 ")):
                sent.append(line.split()[0])
                charges.append(engine.slot_kv_pages[0])
            if line.startswith("BSTOP 0"):
                stopped = True
            send(line)

        def take(*args, **kwargs):
            acquired = yield from take_control(*args, **kwargs)
            if acquired and stopped:
                with engine.slot_cv:
                    if case == "waiter" and not tickets:
                        tickets.append(engine.admission.join(1))
                    elif case == "cancel":
                        cancel.set()
                    elif case == "restart":
                        engine.slot_busy = [True, False]
                        engine.slot_kv_pages = [22, 0]
            return acquired

        def consume():
            for token in engine.generate(list(b"LONGREPLY"), 80, {"seed": 41}, cancel):
                if token is not None:
                    outputs.append(int(token))
                    if len(outputs) == 1:
                        with engine.slot_cv:
                            engine.slot_busy[1] = False
                            engine.slot_kv_pages[1] = 0
                            engine.slot_cv.notify_all()

        try:
            with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1", "BASALT_SHARED_KV_INCREMENTAL": "1"}), \
                    mock.patch.object(engine, "_send", side_effect=record), \
                    mock.patch.object(engine, "_take_control", side_effect=take):
                if case == "restart":
                    with self.assertRaisesRegex(EngineDied, "restarted during shared solo transition"):
                        consume()
                else:
                    consume()
            self.assertEqual(sent[:2], ["BGEN", "BSTOP"])
            if case in ("solo", "waiter"):
                self.assertEqual(sent, ["BGEN", "BSTOP", "SGEN" if case == "solo" else "BGEN"])
                self.assertEqual(outputs, list(b"ok, " + b"la " * 15 + b"done.") + [257])
            else:
                self.assertEqual(len(sent), 2)
                self.assertGreater(len(outputs), 0)
            self.assertEqual(len(set(charges)), 1)
            self.assertGreater(charges[0], 0)
            self.assertEqual(engine.slot_busy, [True, False] if case == "restart" else [False, False])
            self.assertEqual(engine.slot_kv_pages, [22, 0] if case == "restart" else [0, 0])
        finally:
            for ticket in tickets:
                engine.admission.leave(ticket)
            if case == "restart":
                engine.slot_busy, engine.slot_kv_pages = old_busy, old_pages
                old_busy[0], old_pages[0] = False, 0

    def test_shared_pool_admits_other_adapter_while_first_is_active(self):
        self.start(2)
        self.engine.info.update(shared_kv_pages=128, shared_kv_page_cells=4,
                                shared_kv_mixed_adapters=1)
        with self.engine.slot_cv:
            self.engine.slot_busy[0] = True
            self.engine.slot_kv_pages[0] = 32
            self.engine.slot_adapters[0] = False
        tokens = [1, 2, 3]
        with mock.patch.object(self.engine, "_send", wraps=self.engine._send) as send:
            list(self.engine.generate(tokens, 12, {"_atomic_lora": True}, threading.Event()))
        lines = [call.args[0] for call in send.call_args_list]
        self.assertTrue(any(line.startswith("BGEN 1 ") and " atomic_lora=1 " in line
                            for line in lines), lines)
        self.assertEqual(self.engine.slot_kv_pages, [32, 0])
        self.assertTrue(self.engine.slot_busy[0])

    def test_shared_profile_affinity_and_mixed_admission(self):
        self.start(3)
        self.engine.info.update(shared_kv_pages=128, shared_kv_page_cells=4, rope_context_profiles=1)
        with self.engine.slot_cv:
            self.engine.slot_busy[0] = True
            self.engine.slot_kv_pages[0] = 32
            self.engine.slot_held[:] = [[1, 2], [1, 2], [1, 2]]
            self.engine.slot_adapters[:] = [0, 0, 4]
            self.engine.claim_slot(1, "same-session")
        sampling = {"_rope_profile": 1, "_context_limit": 524288}
        with mock.patch.object(self.engine, "_send", wraps=self.engine._send) as send:
            list(self.engine.generate([1, 2, 3], 12, sampling, threading.Event(), session="same-session"))
        lines = [call.args[0] for call in send.call_args_list]
        self.assertTrue(any(line.startswith("BGEN 2 ") and " rope_profile=1 " in line for line in lines), lines)
        self.assertEqual(self.engine.slot_kv_pages, [32, 0, 0])
        self.assertTrue(self.engine.slot_busy[0])

    def test_cancellation_during_shared_growth_keeps_new_slot_idle(self):
        self.start(2)
        self.engine.info.update(shared_kv_pages=128, shared_kv_page_cells=4, shared_kv_grow=1)
        with self.engine.slot_cv:
            self.engine.slot_busy[:] = [True, True]
            self.engine.slot_kv_pages[:] = [20, 20]
        cancel = threading.Event()

        def reply(line, **kwargs):
            self.assertEqual(line, "BGROW")
            cancel.set()
            self.engine.lines.put("BGROWN 2")

        with mock.patch.object(self.engine, "_send", side_effect=reply) as send:
            self.assertEqual(list(self.engine.generate([1, 2, 3], 12, {}, cancel)), [])
            send.assert_called_once_with("BGROW", proc=self.engine.proc)
        self.assertEqual(self.engine.slot_busy, [True, True, False])
        self.assertEqual(self.engine.slot_kv_pages, [20, 20, 0])
        self.assertEqual(len(self.engine.admission), 0)
        self.assertFalse(self.engine.ctl.locked())
        self.assertTrue(list(self.engine.generate([1, 2, 3], 12, {}, threading.Event())))
        self.assertEqual(self.engine.slot_kv_pages, [20, 20, 0])

    def cold_routing_setup(self):
        self.start(1, slot0_solo=True, solo_any=True)
        self.engine.info.update(shared_kv_pages=8192, shared_kv_page_cells=4, shared_kv_grow=1,
                                rope_context_profiles=1, cold_shared_recovery_min_tokens=8192)
        self.engine.slot_adapters[0] = 4
        self.engine.slot_held[0] = [7] * 9000
        self.engine.claim_slot(0, "long-yarn-owner")

    def test_cold_profile_grows_before_replacing_slot_zero(self):
        self.cold_routing_setup()
        send = self.engine._send
        outgoing = self.engine.slot_held[0]

        def reply(line, **kwargs):
            if line == "BGROW 0":
                self.engine.lines.put("BGROWN 1")
            else:
                send(line, **kwargs)

        with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1"}):
            with mock.patch.object(self.engine, "_send", side_effect=reply) as transport:
                self.assertTrue(list(self.engine.generate([1, 2, 3], 12, {"_rope_profile": 0, "_context_limit": 262144}, threading.Event())))
        lines = [call.args[0] for call in transport.call_args_list]
        self.assertEqual(lines[0], "BGROW 0")
        self.assertTrue(any(line.startswith("BGEN 1 ") for line in lines), lines)
        self.assertIs(self.engine.slot_held[0], outgoing)
        self.assertEqual(self.engine.slot_owner[0], "long-yarn-owner")
        self.assertEqual(self.engine.slot_kv_pages, [0, 0])
        self.assertFalse(self.engine.ctl.locked())

    def test_mixed_alias_grows_and_returns_to_warm_slot_zero(self):
        self.cold_routing_setup()
        engine = self.engine
        engine.max_context = 1 << 20
        outgoing = engine.slot_held[0]
        send = engine._send
        swift = {"_rope_profile": 0, "_swift_lora": True, "_context_limit": 262144}
        yarn = {"_rope_profile": 1, "_context_limit": 524288}

        def reply(line, **kwargs):
            if line == "BGROW":
                engine.lines.put("BGROWN 1")
            else:
                send(line, **kwargs)

        with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1"}):
            with mock.patch.object(engine, "_send", side_effect=reply) as transport:
                self.assertTrue(list(engine.generate([1, 2, 3], 12, swift, threading.Event(), session="swift")))
                self.assertIs(engine.slot_held[0], outgoing)
                self.assertEqual(engine.slot_owner[0], "long-yarn-owner")
                self.assertTrue(list(engine.generate(outgoing + [8], 12, yarn, threading.Event(),
                                                     session="long-yarn-owner")))
        lines = [call.args[0] for call in transport.call_args_list]
        self.assertEqual(lines[0], "BGROW")
        self.assertTrue(any(line.startswith("BGEN 1 ") for line in lines))
        self.assertTrue(any(line.startswith("SGEN 0 ") for line in lines))

    def test_mixed_alias_uses_available_peer_without_growth(self):
        self.start(2, slot0_solo=True, solo_any=True)
        engine = self.engine
        engine.info.update(shared_kv_pages=8192, shared_kv_page_cells=4, shared_kv_grow=1,
                           rope_context_profiles=1, cold_shared_recovery_min_tokens=8192)
        engine.slot_adapters[:] = [4, 2]
        outgoing = [7] * 9000
        engine.slot_held[:] = [outgoing, [1, 2]]
        engine.claim_slot(1, "prior-swift")
        swift = {"_rope_profile": 0, "_swift_lora": True, "_context_limit": 262144}
        with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1"}):
            with mock.patch.object(engine, "_send", wraps=engine._send) as transport:
                self.assertTrue(list(engine.generate([1, 2, 3], 12, swift, threading.Event())))
        lines = [call.args[0] for call in transport.call_args_list]
        self.assertTrue(lines[0].startswith("BGEN 1 "))
        self.assertIs(engine.slot_held[0], outgoing)

    def test_mixed_alias_retires_source_when_both_reservations_cannot_fit(self):
        self.cold_routing_setup()
        engine = self.engine
        engine.info["shared_kv_pages"] = 2250
        swift = {"_rope_profile": 0, "_swift_lora": True, "_context_limit": 262144}
        with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1"}):
            with mock.patch.object(engine, "_send", wraps=engine._send) as transport:
                self.assertTrue(list(engine.generate([1, 2, 3], 12, swift, threading.Event())))
        self.assertTrue(transport.call_args_list[0].args[0].startswith("SGEN 0 "))

    def test_second_replacement_prefers_nonzero_slot_while_source_is_frozen(self):
        self.check_second_frozen_replacement(2200, False, 2)

    def test_second_replacement_falls_back_when_frozen_pages_fill_the_pool(self):
        self.check_second_frozen_replacement(8190, False, 0)

    def test_second_replacement_falls_back_when_extra_growth_is_refused(self):
        self.check_second_frozen_replacement(2200, True, 0)

    def check_second_frozen_replacement(self, pages, refuse, expected):
        fake = FAKE_BATCH.replace(
            '    if line.startswith("BSTOP "):',
            '    if line.startswith("BGROW"):\n'
            f'        if slots == 2 and {refuse!r}:\n'
            '            print("BGROWN -1", flush=True)\n'
            '        else:\n'
            '            print(f"BGROWN {slots}", flush=True)\n'
            '            slots += 1\n'
            '        continue\n'
            '    if line.startswith("BSTOP "):')
        fake = fake.replace('    for b in sorted(active):',
                            '    for b in sorted(active):\n        if b == 1 and b not in stopped:\n            continue')
        fake = fake.replace('            print(f"BADM {slot} {1 if cont else 0}", flush=True)',
                            f'            if slot == 1:\n'
                            f'                print("INFO frozen_shared_kv_pages={pages} frozen_shared_kv_cache=4", flush=True)\n'
                            '            print(f"BADM {slot} {1 if cont else 0}", flush=True)')
        with mock.patch(__name__ + ".FAKE_BATCH", fake):
            self.cold_routing_setup()

        engine = self.engine
        seed = {"_rope_profile": 0, "_context_limit": 262144}
        cancel = threading.Event()
        first = threading.Event()
        errors = []

        def generate():
            try:
                for token in engine.generate(list(b"LONGREPLY"), 64, seed, cancel, session="B"):
                    if token is not None:
                        first.set()
            except Exception as err:
                errors.append(err)
                first.set()

        with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1"}):
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            try:
                self.assertTrue(first.wait(3))
                self.assertFalse(errors)
                self.assertTrue(engine.slot_busy[1])
                self.assertTrue(list(engine.generate([1, 2, 3], 16, seed, threading.Event(), session="C")))
                self.assertTrue(engine.slot_busy[1])
                admitted = [int(row.split()[1]) for row in self.log.read_text().splitlines() if row.startswith("BGEN ")]
                self.assertEqual(admitted, [1, expected])
            finally:
                cancel.set()
                engine._send("BSTOP 1")
                thread.join(3)
        self.assertFalse(errors)
        self.assertFalse(thread.is_alive())

    def test_cold_growth_refusal_reuses_original_slot(self):
        self.cold_routing_setup()
        send = self.engine._send

        def reply(line, **kwargs):
            if line == "BGROW 0":
                self.engine.lines.put("BGROWN -1")
            else:
                send(line, **kwargs)

        with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1"}):
            with mock.patch.object(self.engine, "_send", side_effect=reply) as transport:
                self.assertTrue(list(self.engine.generate([1, 2, 3], 12, {"_rope_profile": 0, "_context_limit": 262144}, threading.Event())))
        lines = [call.args[0] for call in transport.call_args_list]
        self.assertEqual(lines[0], "BGROW 0")
        self.assertTrue(any(line.startswith("BGEN 0 ") for line in lines), lines)
        self.assertEqual(self.engine.slot_kv_pages, [0])
        self.assertEqual(len(self.engine.admission), 0)

    def test_cold_growth_accounts_for_outgoing_prefix(self):
        self.cold_routing_setup()
        self.engine.info["shared_kv_pages"] = 2250
        with mock.patch.dict(os.environ, {"BASALT_SHARED_SOLO": "1"}):
            with mock.patch.object(self.engine, "_send", wraps=self.engine._send) as transport:
                self.assertTrue(list(self.engine.generate([1, 2, 3], 12, {"_rope_profile": 0, "_context_limit": 262144}, threading.Event())))
        lines = [call.args[0] for call in transport.call_args_list]
        self.assertFalse(any(line.startswith("BGROW") for line in lines), lines)
        self.assertTrue(any(line.startswith("SGEN 0 ") for line in lines), lines)

    def test_shared_growth_does_not_apply_reply_from_previous_generation(self):
        self.cold_routing_setup()
        previous_lines = self.engine.lines

        def reply(line, **kwargs):
            self.assertEqual(line, "BGROW 0")
            self.engine.gen += 1
            self.engine.lines = queue.Queue()
            previous_lines.put("BGROWN 1")

        with mock.patch.object(self.engine, "_send", side_effect=reply):
            with self.assertRaisesRegex(EngineDied, "restarted during shared slot growth"):
                self.engine._grow_shared_slot(0)
        self.assertEqual(self.engine.slot_busy, [False])
        self.assertEqual(self.engine.slot_kv_pages, [0])

    def test_restart_drops_unadvertised_cold_growth_capability(self):
        self.cold_routing_setup()
        script = Path(self.tmp.name) / "fake_engine.py"
        real = subprocess.Popen
        with mock.patch.object(subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine.restart(tries=1)
        self.assertNotIn("cold_shared_recovery_min_tokens", self.engine.info)

    def test_shared_growth_refusal_waits_for_existing_slot(self):
        self.start(2)
        self.engine.info.update(shared_kv_pages=128, shared_kv_page_cells=4, shared_kv_grow=1)
        with self.engine.slot_cv:
            self.engine.slot_busy[:] = [True, True]
            self.engine.slot_kv_pages[:] = [20, 20]
        result = []
        refused = threading.Event()
        send = self.engine._send

        def reply(line, **kwargs):
            if line == "BGROW":
                self.engine.lines.put("BGROWN -1")
                refused.set()
            else:
                send(line, **kwargs)

        thread = threading.Thread(target=lambda: result.append(self.chat("wait after growth refusal")))
        with mock.patch.object(self.engine, "_send", side_effect=reply) as transport:
            thread.start()
            try:
                self.assertTrue(refused.wait(5))
                with self.engine.slot_cv:
                    self.assertEqual(len(self.engine.admission), 1)
                    self.assertEqual(self.engine.slot_kv_pages, [20, 20])
                    self.assertFalse(self.engine.ctl.locked())
                transport.assert_called_once_with("BGROW", proc=self.engine.proc)
            finally:
                with self.engine.slot_cv:
                    self.engine.slot_busy[1] = False
                    self.engine.slot_kv_pages[1] = 0
                    self.engine.slot_cv.notify_all()
                thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.engine.slot_kv_pages, [20, 0])

    def test_shared_pool_waits_even_when_a_slot_is_free(self):
        self.start(2)
        self.engine.info.update(shared_kv_pages=128, shared_kv_page_cells=4)
        with self.engine.slot_cv:
            self.engine.slot_busy[0] = True
            self.engine.slot_kv_pages[0] = 127
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("r", self.chat("waits for pages")))
        thread.start()
        try:
            end = time.monotonic() + 5
            while time.monotonic() < end and not len(self.engine.admission):
                time.sleep(0.01)
            self.assertEqual(len(self.engine.admission), 1)
            self.assertFalse(self.engine.slot_busy[1])
            self.assertFalse(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        finally:
            with self.engine.slot_cv:
                self.engine.slot_busy[0] = False
                self.engine.slot_kv_pages[0] = 0
                self.engine.slot_cv.notify_all()
            thread.join(15)
        self.assertEqual(result["r"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.engine.slot_kv_pages, [0, 0])

    def test_reads_that_gave_way_in_every_slot_do_not_stall(self):
        """Two long reads, then short requests: no read gives way to a request that finds no free slot, and
        nothing stalls with every slot holding a read that gave way (issue #2)."""
        self.start(2)
        out, errors = {}, []

        def go(t):
            try:
                out[t] = self.chat(t, max_tokens=16)["choices"][0]["message"]["content"]
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)
        texts = [("aaaa " * 600, 0.0), ("bbbb " * 700, 0.05), ("short one", 0.6), ("short two", 0.1)]
        threads = []
        for t, delay in texts:
            time.sleep(delay)
            threads.append(threading.Thread(target=go, args=(t,)))
            threads[-1].start()
        for th in threads:
            th.join(20)
        self.assertFalse(any(th.is_alive() for th in threads))
        self.assertEqual(errors, [])
        self.assertEqual(sorted(out.values()), ["ok, done."] * 4)
        self.assertFalse(any(self.engine.slot_busy))

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request left alone in a slot returns to the solo path (needs INFO slot_cache=1)."""
        self.start(2, slot_cache=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(res["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")

    def test_a_failed_window_names_its_error(self):
        # a batch window fails while every request reads its own slot's lines, so no request reads the engine's ERR
        # line before it exits; the death note gives it instead of a guess
        self.start(2, fail=True)
        errors = []

        def go(t):
            try:
                self.chat(t)
            except urllib.error.HTTPError as e:
                errors.append(e.code)

        threads = [threading.Thread(target=go, args=(t,)) for t in ("first question LONGREPLY", "second question")]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(len(errors), 2, errors)
        note = self.engine.death_note()
        self.assertIn("exited (code 1)", note)
        self.assertIn("verify batch: layer 34 never rang (graph finished)", note)

    def test_stop_strings_in_a_batch_slot(self):
        """A stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"model": "basalt", "messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def several(self, multi):
        self.start(2, multi=multi)
        texts = ["first question LONGREPLY", "second question, longer"]
        out = {}
        threads = [threading.Thread(target=lambda t=t: out.setdefault(t, self.chat(t))) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(out[texts[0]]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertEqual(out[texts[1]]["choices"][0]["message"]["content"], "ok, done.")

    def test_a_window_gives_several_tokens_in_one_line(self):
        self.several(4)

    def test_a_window_gives_one_token_in_a_line(self):
        self.several(1)

    def slot_drafts(self):
        return {x["id"]: (x["drafts_accepted"], x["drafts_offered"]) for x in self.get("/slots")}

    def test_batch_slot_drafts_come_from_its_bdone(self):
        # BDONE <slot> <n> <fin> <ms> <accepted> <offered>: each slot keeps its own counts
        self.start(2, drafts=True)
        th = threading.Thread(target=lambda: self.chat("blocker LONGREPLY", max_tokens=200))
        th.start()
        time.sleep(0.05)
        self.chat("second")
        th.join(30)
        got = self.slot_drafts()
        self.assertEqual(got, {0: (3, 10), 1: (4, 20)})

    def timed_chats(self, texts, gap=0.05, max_tokens=200):
        """Each text as a streamed request started `gap` s after the previous; the `timings` of every reply."""
        out = [None] * len(texts)

        def go(i):
            body = {"model": "basalt", "messages": [{"role": "user", "content": texts[i]}], "max_tokens": max_tokens,
                    "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        j = json.loads(line[5:])
                        if j.get("timings"):
                            out[i] = j["timings"]

        threads = [threading.Thread(target=go, args=(i,)) for i in range(len(texts))]
        for th in threads:
            th.start()
            time.sleep(gap)
        for th in threads:
            th.join(30)
        return out

    def usages(self, texts, gap=0.05, max_tokens=200):
        """Each text as a streamed request started `gap` s after the previous; the `usage` of every reply."""
        out = [None] * len(texts)

        def go(i):
            body = {"model": "basalt", "messages": [{"role": "user", "content": texts[i]}], "max_tokens": max_tokens,
                    "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        j = json.loads(line[5:])
                        if j.get("usage"):
                            out[i] = j["usage"]

        threads = [threading.Thread(target=go, args=(i,)) for i in range(len(texts))]
        for th in threads:
            th.start()
            time.sleep(gap)
        for th in threads:
            th.join(30)
        return out

    def test_cached_tokens_are_the_prompt_found_in_an_earlier_cache(self):
        self.start(2, reuse=True)
        text = "a long enough question to share " + "z" * 100
        cold = self.usages([text])[0]
        self.assertEqual(cold["prompt_tokens_details"]["cached_tokens"], 0)        # nothing cached before it
        again = self.usages([text])[0]
        cached = again["prompt_tokens_details"]["cached_tokens"]
        self.assertGreater(cached, 0)
        self.assertLessEqual(cached, again["prompt_tokens"])

    def test_tokens_a_request_generates_are_not_cached_for_it(self):
        self.start(2, reuse=True, counts=True)
        # the blocker starts solo and moves to a slot: its second run finds its own prompt and tokens in the slot
        got = self.usages(["blocker LONGREPLY " + "x" * 100, "second one"], gap=0.1, max_tokens=400)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)
        for u in got:
            self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], 0, u)
            self.assertGreater(u["completion_tokens"], 1)

    def test_cached_tokens_never_pass_the_prompt_when_a_slot_extends_it(self):
        self.start(2, reuse=True, counts=True, slot_cache=True)
        text = "blocker LONGREPLY " + "x" * 100
        self.usages([text, "second one"], gap=0.1, max_tokens=400)
        got = self.usages([text, "another"], gap=0.1, max_tokens=400)
        u = got[0]
        self.assertLessEqual(u["prompt_tokens_details"]["cached_tokens"], u["prompt_tokens"], u)

    def convo(self, msgs, text):
        """One more turn of the conversation `msgs`: its usage; the reply is added to `msgs`."""
        msgs.append({"role": "user", "content": text})
        body = {"model": "basalt", "messages": msgs, "max_tokens": 16, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read().decode())
        msgs.append({"role": "assistant", "content": j["choices"][0]["message"]["content"]})
        return j["usage"]

    def kept(self, u, before):
        """Whether a turn reused the previous turn's whole prompt, but for a short tail."""
        return u["prompt_tokens_details"]["cached_tokens"] >= before - 40

    def test_alternating_conversations_each_keep_their_slot_when_slot_0_is_the_solo_session(self):
        self.start(2, reuse=True, slot_cache=True, slot0_solo=True, ready=200000)
        a = [{"role": "user", "content": "conversation A " + "a" * 70000}, {"role": "assistant", "content": "ok, done."}]
        b = [{"role": "user", "content": "conversation B " + "b" * 60000}, {"role": "assistant", "content": "ok, done."}]
        sizes = {"a": self.convo(a, "first")["prompt_tokens"], "b": self.convo(b, "first")["prompt_tokens"]}
        for turn in range(4):                    # every request alone: they alternate
            for name, msgs in (("a", a), ("b", b)):
                u = self.convo(msgs, f"turn {turn} " + name * 300)
                self.assertTrue(self.kept(u, sizes[name]), (name, turn, u, sizes[name]))
                sizes[name] = u["prompt_tokens"]
        for turn in range(3):                    # one conversation alone for a while
            u = self.convo(a, f"alone {turn}")
            self.assertTrue(self.kept(u, sizes["a"]), (turn, u, sizes["a"]))
            sizes["a"] = u["prompt_tokens"]
        u = self.convo(b, "back")                # and the other one is still where it was
        self.assertTrue(self.kept(u, sizes["b"]), (u, sizes["b"]))
        sizes["b"] = u["prompt_tokens"]
        kinds = {x.split()[0] for x in self.log.read_text().splitlines() if not x.startswith("YIELD")}
        self.assertEqual(kinds, {"BGEN"})        # a lone request is a batch of one: no solo run copies into slot 0

    def test_a_lone_request_runs_solo_in_the_slot_holding_its_conversation_when_the_engine_can(self):
        self.start(2, reuse=True, slot_cache=True, slot0_solo=True, solo_any=True, ready=200000)
        a = [{"role": "user", "content": "conversation A " + "a" * 70000}, {"role": "assistant", "content": "ok, done."}]
        b = [{"role": "user", "content": "conversation B " + "b" * 60000}, {"role": "assistant", "content": "ok, done."}]
        sizes = {"a": self.convo(a, "first")["prompt_tokens"], "b": self.convo(b, "first")["prompt_tokens"]}
        for turn in range(3):                    # every request alone: they alternate
            for name, msgs in (("a", a), ("b", b)):
                u = self.convo(msgs, f"turn {turn} " + name * 300)
                self.assertTrue(self.kept(u, sizes[name]), (name, turn, u, sizes[name]))
                sizes[name] = u["prompt_tokens"]
        reqs = [x.split() for x in self.log.read_text().splitlines() if not x.startswith("YIELD")]
        self.assertEqual({r[0] for r in reqs}, {"SGEN"})          # solo runs, in place: no batch of one
        slots = [r[1] for r in reqs]
        self.assertEqual(sorted(set(slots)), ["0", "1"])
        self.assertEqual(slots[2:], slots[:2] * 3, slots)         # each conversation returns to its own slot

    def test_a_third_conversation_evicts_the_least_recently_used_slot(self):
        self.start(2, reuse=True, slot_cache=True, slot0_solo=True, ready=200000)
        a = [{"role": "user", "content": "conversation A " + "a" * 20000}, {"role": "assistant", "content": "ok, done."}]
        b = [{"role": "user", "content": "conversation B " + "b" * 20000}, {"role": "assistant", "content": "ok, done."}]
        c = [{"role": "user", "content": "conversation C " + "c" * 20000}, {"role": "assistant", "content": "ok, done."}]
        sizes = [self.convo(m, "first")["prompt_tokens"] for m in (a, b)]
        sizes[0] = self.convo(a, "again")["prompt_tokens"]              # A is now the most recently used
        self.assertLess(self.convo(c, "first")["prompt_tokens_details"]["cached_tokens"], 100)   # (the template's head)
        u = self.convo(a, "after C")
        self.assertTrue(self.kept(u, sizes[0]), (u, sizes))             # A kept its slot
        u = self.convo(b, "after C")                                    # B was the one evicted
        self.assertLess(u["prompt_tokens_details"]["cached_tokens"], 100, u)

    def check_counts(self, timings, batch_admissions):
        # the fake drafts 2 per accepted; the admission's own token is nobody's draft
        for t in timings:
            self.assertIsNotNone(t)
            self.assertEqual(t["draft_n_accepted"], t["predicted_n"] - batch_admissions, t)
            self.assertEqual(t["draft_n"], 2 * t["draft_n_accepted"], t)

    # ---- session affinity: the header picks the slot -----------------------------------------------------------------
    def chat_session(self, text, headers=None, max_tokens=64):
        """A chat completion carrying a session header (the flow id a client or its load balancer already sends)."""
        body = {"model": "basalt", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def together(self, jobs):
        """Run every job at the same instant: a request alone on an idle engine runs solo without choosing a slot, so
        a slot is only picked (and a session only claimed) when requests overlap."""
        barrier, out, errors = threading.Barrier(len(jobs)), [], []

        def go(job):
            try:
                barrier.wait(10)
                out.append(job())
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)
        threads = [threading.Thread(target=go, args=(j,)) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        return out

    def test_the_same_session_lands_in_the_same_slot(self):
        self.start(3)
        self.together([lambda: self.chat_session("alpha " * 10, {"x-session-id": "s1"}),
                       lambda: self.chat_session("beta " * 14, {"x-session-id": "s2"})])
        first = dict(self.engine.session_slot)
        self.assertEqual(set(first), {"s1", "s2"})                  # the headers reached the engine
        self.assertNotEqual(first["s1"], first["s2"])               # two conversations, two slots
        # the next turn of each session, with a prompt of its own: both land in the slot their K/V sits in, although
        # the engine has a third, unowned slot free (which is what admission would otherwise prefer)
        self.together([lambda: self.chat_session("gamma " * 9, {"x-session-id": "s1"}),
                       lambda: self.chat_session("delta " * 12, {"x-session-id": "s2"})])
        self.assertEqual(self.engine.session_slot, first)
        self.assertEqual(sorted(self.engine.slot_owner.items(), key=lambda kv: kv[0]),
                         sorted(((s, k) for k, s in first.items())))
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_per_request_id_is_not_a_session(self):
        """x-client-request-id names one request, so pinning a slot with it would give every turn a slot of its own."""
        self.start(3)
        self.together([lambda: self.chat_session("epsilon " * 10, {"x-client-request-id": "req-1"}),
                       lambda: self.chat_session("zeta " * 11, {"x-client-request-id": "req-2"})])
        self.assertEqual(self.engine.session_slot, {})
        self.assertEqual(self.engine.slot_owner, {})

    def test_the_session_headers_a_load_balancer_sends_all_count(self):
        self.start(4)
        self.together([(lambda h=h, n=n: self.chat_session("turn " * n, {h: "s-" + h}))
                       for n, h in enumerate(("x-llm-flow-id", "x-claude-code-session-id", "x-session-id",
                                              "x-session-affinity"))])
        for h in ("x-llm-flow-id", "x-claude-code-session-id", "x-session-id", "x-session-affinity"):
            self.assertIn("s-" + h, self.engine.session_slot, h)

    def test_overlapping_requests_of_one_session_spread_over_slots(self):
        """A session's slot is busy with its first request: the second takes the normal pick and owns the slot after."""
        self.start(3)
        out = self.together([lambda: self.chat_session("alpha " * 10, {"x-claude-code-session-id": "s1"}),
                             lambda: self.chat_session("beta " * 14, {"x-claude-code-session-id": "s1"})])
        self.assertEqual(len(out), 2)                                # neither waits on the other: both complete
        for r in out:
            self.assertEqual(r["choices"][0]["message"]["content"], "ok, done.")
        slots = [int(x.split()[1]) for x in self.log.read_text().splitlines() if x.startswith(("BGEN", "SGEN"))]
        self.assertEqual(len(set(slots)), 2, slots)                  # two slots, not one request queued behind the other
        self.assertEqual(list(self.engine.session_slot), ["s1"])     # one slot owned by the session, the later claim
        self.assertEqual(self.engine.slot_owner, {self.engine.session_slot["s1"]: "s1"})
        self.assertFalse(any(self.engine.slot_busy))                 # nothing is left held
        # the next turn goes back to the slot it owns
        self.chat_session("gamma " * 9, {"x-claude-code-session-id": "s1"})
        self.assertEqual(len(self.engine.session_slot), 1)

    def test_timings_of_a_solo_request_carry_its_draft_counts(self):
        self.start(2, counts=True)
        self.check_counts(self.timed_chats(["alone LONGREPLY"]), 0)

    def test_timings_of_requests_in_batch_slots_carry_their_own_draft_counts(self):
        self.start(2, counts=True)
        # the second request arrives while the first reads its prompt, so both run in slots
        self.check_counts(self.timed_chats(["first LONGREPLY " + "x" * 400, "second LONGREPLY " + "y" * 400], gap=0), 1)

    def test_timings_of_a_request_that_moves_from_solo_to_a_slot_sum_both_parts(self):
        self.start(2, counts=True)
        got = self.timed_chats(["blocker LONGREPLY", "second one"], gap=0.1)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)          # the blocker did start solo and moved
        self.check_counts(got[:1], 1)
        self.check_counts(got[1:], 1)

    def test_timings_of_a_request_that_returns_from_a_slot_to_solo_sum_all_parts(self):
        self.start(2, counts=True, slot_cache=True)
        got = self.timed_chats(["blocker LONGREPLY", "second one"], gap=0.1, max_tokens=400)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds, ["GEN", "BGEN", "BGEN", "GEN"], kinds)    # solo, a slot (and the other's), solo again
        self.check_counts(got[:1], 1)
        self.check_counts(got[1:], 1)

    def test_the_prompt_time_of_a_request_that_moves_between_solo_and_slots_is_its_read_not_the_last_run(self):
        """A request that goes solo -> slot -> solo again reads its prompt once; the runs after the first re-read one
        token, and their ~0 ms must not become the request's prompt time (a huge tok/s in the done line)."""
        from serve.service import done_line
        self.start(2, counts=True, slot_cache=True, pms=True)
        got = self.timed_chats(["blocker LONGREPLY", "second one"], gap=0.1, max_tokens=400)
        kinds = [l.split()[0] for l in self.log.read_text().splitlines()]
        self.assertEqual(kinds, ["GEN", "BGEN", "BGEN", "GEN"], kinds)
        t = got[0]
        self.assertGreaterEqual(t["prompt_ms"], 500.0)
        self.assertLess(t["prompt_ms"], 502.0)
        self.assertLess(t["prompt_per_second"], t["prompt_n"] * 2.0)         # n tokens in ~0.5 s
        self.assertIn(" 0.5s", done_line(0, t["prompt_n"], 1, "stop", False, t, 0.5, 1.0))
        self.assertEqual(got[1]["prompt_ms"], 500.0)                          # the other request read its own once

    def test_old_bdone_without_counts_keeps_drafts_unknown(self):
        self.start(2)
        th = threading.Thread(target=lambda: self.chat("blocker LONGREPLY", max_tokens=200))
        th.start()
        time.sleep(0.05)
        self.chat("second")
        th.join(30)
        self.assertEqual(self.slot_drafts(), {0: (None, None), 1: (None, None)})

    def test_context_is_what_each_slot_holds(self):
        # the engine's context is the slots' own (the server divides --max-context): /slots, /props and /v1/models
        # say it, and the 400 for a prompt over it names it
        self.start(3, ready=1024)
        self.assertEqual([x["n_ctx"] for x in self.get("/slots")], [1024, 1024, 1024])
        self.assertEqual(self.get("/props")["default_generation_settings"]["n_ctx"], 1024)
        self.assertEqual(self.get("/v1/models")["data"][0]["meta"]["n_ctx"], 1024)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.chat("long " * 1100)
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("(1024)", cm.exception.read().decode())


class SessionAffinity(PickSlot):
    """A session header keeps a conversation in its own slot, so the next turn continues the K/V already there."""

    def test_a_repeated_session_takes_its_own_slot(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3, 4], [9, 9], [1, 2]]
        e.slot_used = [0.0, 1.0, 2.0]
        # the longest held prefix would be slot 0; the session owns slot 1 and that wins
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], session="s1"), 0)
        e.claim_slot(1, "s1")
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], session="s1"), 1)
        self.assertEqual(e.session_slot, {"s1": 1})
        self.assertEqual(e.slot_owner, {1: "s1"})

    def test_another_sessions_slot_is_taken_only_when_no_unowned_one_is_free(self):
        e = self.engine(3)
        e.claim_slot(0, "s1")
        e.claim_slot(1, "s2")
        # slot 2 is unowned, so a third session gets it rather than evicting anyone
        self.assertEqual(e.pick_slot([7], session="s3"), 2)
        # with no unowned slot free, a session takes another's - the least recently used of them
        e.slot_busy = [True, True, True]
        self.assertIsNone(e.pick_slot([7], session="s3"))
        e.slot_busy = [False, False, True]
        e.slot_used = [5.0, 1.0, 0.0]
        self.assertEqual(e.pick_slot([7], session="s3"), 1)   # slot 1 is the least recently used of {0, 1}
        e.claim_slot(1, "s3")
        # s2 lost its slot and s3 owns it now; one slot, one owner
        self.assertNotIn("s2", e.session_slot)
        self.assertEqual(e.slot_owner, {0: "s1", 1: "s3"})

    def test_claiming_frees_the_sessions_previous_slot(self):
        e = self.engine(3)
        e.claim_slot(0, "s1")
        e.claim_slot(2, "s1")            # the same session moved
        self.assertEqual(e.session_slot, {"s1": 2})
        self.assertEqual(e.slot_owner, {2: "s1"})   # slot 0 is free of the stale claim

    def test_no_session_behaves_exactly_as_before(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        before = [e.pick_slot(p) for p in ([1, 2, 3, 4, 5, 6], [1, 2, 3, 9], [7, 8], [1, 2, 3])]
        e.claim_slot(0, None)            # a claim with no session is a no-op
        self.assertEqual(e.slot_owner, {})
        self.assertEqual(e.session_slot, {})
        after = [e.pick_slot(p, None) for p in ([1, 2, 3, 4, 5, 6], [1, 2, 3, 9], [7, 8], [1, 2, 3])]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
