"""Buffer a candidate repeated output run before native tokens reach the API."""
from functools import lru_cache

from serve.engine.errors import EngineDied


class OutputLoop(EngineDied):
    """The native worker emitted too many tokens in an unbroken repeated run."""


def checked_tokens(tokens, stop_ids, decode, limit=256):
    @lru_cache(maxsize=1024)
    def classify(token):
        piece = decode([token])
        return piece[0] if piece and piece == piece[0] * len(piece) else ""

    pending = []
    last_character = ""
    for token in tokens:
        if token is None:
            yield None
            continue
        eligible = token not in stop_ids
        character = classify(int(token)) if eligible else ""
        if pending and (not eligible or (token != pending[-1] and not (character and character == last_character))):
            yield from pending
            pending.clear()
        last_character = character
        if not eligible:
            yield token
            continue
        pending.append(token)
        if len(pending) >= limit:
            raise OutputLoop(f"the native engine generated a repeated output run of {limit} tokens")
    yield from pending
