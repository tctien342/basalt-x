"""Keep bounded, independently validated overlap observations across worker starts."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time

from serve.engine.overlap_hint import SCOPE, TTL_S, fingerprint

ENV = "BASALT_NATIVE_OVERLAP_LEARNING"
DIRECTORY = "BASALT_OVERLAP_LEARN_DIR"
MAX_SCOPES = 16
MAX_BYTES = 8192


def valid_chunk(chunk):
    return type(chunk) is int and 256 <= chunk <= 32768 and chunk % 256 == 0


class OverlapLearning:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.rows = self._read()
        self.writer = None
        self.pending = None
        self.write_failed = False
        self.last_outcome = None

    def _read(self):
        try:
            with self.path.open("rb") as stream:
                payload = stream.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                return {}
            doc = json.loads(payload)
            entries = doc.get("entries")
            if doc.get("version") != 1 or not isinstance(entries, dict) or len(entries) > MAX_SCOPES:
                return {}
            return self._fresh(entries)
        except (OSError, TypeError, ValueError, AttributeError):
            return {}

    @staticmethod
    def _fresh(entries):
        now = int(time.time())
        return {key: row for key, row in entries.items()
                if SCOPE.fullmatch(key) and int(key, 16) and isinstance(row, dict)
                and set(row) == {"chunk", "wins", "verified_at"} and valid_chunk(row["chunk"])
                and type(row["wins"]) is int and row["wins"] in (1, 2)
                and type(row["verified_at"]) is int and 0 <= now - row["verified_at"] < TTL_S}

    def startup_env(self):
        with self.lock:
            entries = self._fresh(self.rows)
            return ";".join(f"{key},{row['chunk']},{row['verified_at'] + TTL_S}"
                            for key, row in entries.items() if row["wins"] == 2)

    def snapshot(self):
        with self.lock:
            entries = self._fresh(self.rows)
            qualified = sum(row["wins"] == 2 for row in entries.values())
            return {
                "qualified_scopes": qualified,
                "provisional_scopes": len(entries) - qualified,
                "scope_limit": MAX_SCOPES,
                "store_state": "pending" if self.writer is not None else "error" if self.write_failed else "synced",
                "last_outcome": dict(self.last_outcome) if self.last_outcome is not None else None,
            }

    def observe(self, scope, chunk, won):
        if not SCOPE.fullmatch(scope) or not int(scope, 16) or not valid_chunk(chunk):
            return
        with self.lock:
            self.rows = self._fresh(self.rows)
            prior = self.rows.get(scope, {})
            same = prior.get("chunk") == chunk
            now = int(time.time())
            if won:
                phase = "revalidated" if same and prior["wins"] == 2 else "qualified" if same else "provisional"
            else:
                phase = "revoked" if same else "rejected"
            self.last_outcome = {"outcome": phase, "scope": scope, "chunk_tokens": chunk, "at": now}
            if won:
                if same and prior["wins"] == 2:
                    return
                self.rows[scope] = {"chunk": chunk, "wins": prior["wins"] + 1 if same else 1,
                                    "verified_at": now}
                if len(self.rows) > MAX_SCOPES:
                    oldest = min((key for key in self.rows if key != scope),
                                 key=lambda key: (self.rows[key]["verified_at"], key))
                    del self.rows[oldest]
            elif same:
                del self.rows[scope]
            else:
                return
            self.pending = self.rows.copy()
            if self.writer is None:
                self.writer = threading.Thread(target=self._write_pending, daemon=True)
                self.writer.start()

    def _write_pending(self):
        while True:
            with self.lock:
                rows, self.pending = self.pending, None
                if rows is None:
                    self.writer = None
                    return
            try:
                self._write(rows)
            except OSError as error:
                with self.lock:
                    self.write_failed = True
                print(f"[server] overlap learning write failed: {error}", flush=True)
            else:
                with self.lock:
                    self.write_failed = False

    def _write(self, rows):
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
            part = Path(stream.name)
            try:
                json.dump({"version": 1, "entries": rows}, stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(part, self.path)
                directory = os.open(part.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                part.unlink(missing_ok=True)


def load(engine_bin, args, cwd, env, current):
    source = os.environ if env is None else env
    if not any(key in source for key in (ENV, DIRECTORY)):
        return env, None
    result = dict(source)
    result.pop(ENV, None)
    name = result.pop(DIRECTORY, "")
    if not name or os.name != "posix" or "BASALT_NATIVE_OVERLAP_HINT" in result:
        return result, None
    directory = Path(name)
    if not directory.is_absolute() or not directory.is_dir():
        return result, None
    try:
        path = directory / (fingerprint(engine_bin, args, cwd, result) + ".json")
    except OSError:
        return result, None
    state = current if current and current.path == path else OverlapLearning(path)
    result[ENV] = state.startup_env()
    return result, state
