"""A bounded FIFO with one resident CPU speech recognizer."""
import os
from pathlib import Path
from array import array
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
import subprocess
from tempfile import NamedTemporaryFile

from serve.engine.errors import EngineError


class SpeechBusy(Exception):
    """The running job and waiting jobs already fill the admission budget."""


def pin_worker(threads):
    """Use one allowed logical CPU per physical core before filling SMT siblings."""
    if not hasattr(os, "sched_getaffinity"):
        return
    available = sorted(os.sched_getaffinity(0))
    physical = {}
    for cpu in available:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            siblings = topology.read_text().strip()
        except OSError:
            siblings = str(cpu)
        physical.setdefault(siblings, cpu)
    preferred = list(physical.values())
    preferred.extend(cpu for cpu in available if cpu not in preferred)
    os.sched_setaffinity(0, preferred[:threads])


class SpeechWorker:
    def __init__(self, model, threads=8, waiting=4):
        self.model_path = model
        self.threads = threads
        self._slots = BoundedSemaphore(1 + waiting)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="basalt-stt",
                                            initializer=pin_worker, initargs=(threads,))
        self._media_lock = Lock()
        self._video_bytes = 0
        self._model = None
        self._session = None

    def retain_video(self, size, limit):
        with self._media_lock:
            if self._video_bytes + size > limit:
                raise SpeechBusy("too many video bytes awaiting inference; try again shortly")
            self._video_bytes += size

    def release_video(self, size):
        with self._media_lock:
            self._video_bytes -= size

    def submit(self, prepare, language="vi-VN"):
        """Admit before reading media. prepare returns 16 kHz mono float PCM bytes."""
        if not self._slots.acquire(blocking=False):
            raise SpeechBusy("speech transcription queue is full; try again shortly")
        try:
            job = self._executor.submit(self._run, prepare, language)
        except BaseException:
            self._slots.release()
            raise
        job.add_done_callback(lambda _: self._slots.release())
        return job

    def _run(self, prepare, language):
        raw = prepare()
        if not raw:
            return ""
        try:
            if self._session is None:
                from transcribe_cpp import Model

                model = Model(self.model_path, backend="cpu")
                try:
                    self._session = model.session(n_threads=self.threads)
                except Exception:
                    model.close()
                    raise
                self._model = model
            pcm = array("f")
            pcm.frombytes(raw)
            return transcribe_timeline(self._session, pcm, language)
        except Exception as exc:
            print(f"[server] STT worker error: {exc}", flush=True)
            raise EngineError("speech transcription failed") from None

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._session is not None:
            self._session.close()
            self._model.close()


def transcribe_timeline(session, pcm, language):
    rate = 16000
    window = 30 * rate
    lines = []
    for offset in range(0, len(pcm), window):
        chunk = pcm[offset:offset + window]
        result = session.run(chunk, language=language, timestamps="segment")
        text = result.text.strip()
        if not text:
            continue
        if len(pcm) <= window:
            return "\n".join(
                f"[{segment.t0_ms / 1000:.2f}s–{segment.t1_ms / 1000:.2f}s] {segment.text}"
                for segment in result.segments
            ) or text
        start, end = offset / rate, (offset + len(chunk)) / rate
        lines.append(f"[{start:.2f}s–{end:.2f}s] {text}")
    if not lines:
        return ""
    return (f"Audio duration: {len(pcm) / rate:.2f} seconds. Each bracket gives seconds from the start "
            "of the original media, covering the indicated audio window. Use these original times when citing speech.\n"
            + "\n".join(lines))


_MAX_SECONDS = 600
_PCM_BYTES_PER_SECOND = 16000 * 4
_DEMUXERS = "aac,avi,flac,matroska,mov,mp3,mpeg,mpegts,ogg,wav,webm"


def decode_audio(data, *, video=False):
    """Decode the first audio stream and bound duration and subprocess time."""
    with NamedTemporaryFile() as media:
        media.write(data)
        media.flush()
        try:
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                 "-format_whitelist", _DEMUXERS, "-i", media.name, "-map", "0:a:0",
                 "-t", str(_MAX_SECONDS + 1), "-vn", "-ar", "16000", "-ac", "1",
                 "-f", "f32le", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise ValueError("audio decoding timed out") from None
    if result.returncode:
        if video and b"matches no streams" in result.stderr:
            return b""
        raise ValueError("could not decode the first audio track")
    if len(result.stdout) > _MAX_SECONDS * _PCM_BYTES_PER_SECOND:
        raise ValueError(f"audio exceeds {_MAX_SECONDS} seconds")
    return result.stdout
