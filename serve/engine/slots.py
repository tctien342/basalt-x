"""First-come admission to the batch slots, with bounded overtaking."""
from __future__ import annotations

import itertools
import time

OVERTAKE_MAX = 4        # times a waiting request may be passed by later ones before it must go next


class Waiter:
    __slots__ = ("ticket", "skipped", "pages", "adapter", "minimum_pages", "previous_owners",
                 "enqueued", "parked")

    def __init__(self, ticket: int, pages: int, adapter: int, minimum_pages: int | None,
                 previous_owners: frozenset, parked: bool):
        self.ticket, self.skipped = ticket, 0
        self.pages, self.adapter = pages, adapter
        self.minimum_pages = pages if minimum_pages is None else minimum_pages
        self.previous_owners = previous_owners
        self.enqueued, self.parked = time.monotonic(), parked


class AdmissionQueue:
    """Requests waiting for a free slot, oldest first. The oldest takes the next free slot; a later one may go
    first only when `hit` says its prompt resumes a slot's held prefix, and not past the oldest's overtake budget.
    All calls are made under the caller's lock."""

    def __init__(self, overtake_max: int = OVERTAKE_MAX):
        self.overtake_max, self.waiters, self._tickets = overtake_max, [], itertools.count()

    def join(self, pages: int = 0, adapter: int = 0, minimum_pages: int | None = None,
             previous_owners: frozenset = frozenset(), *, parked: bool = False) -> Waiter:
        w = Waiter(next(self._tickets), pages, adapter, minimum_pages, previous_owners, parked)
        self.waiters.append(w)
        return w

    def leave(self, w: Waiter) -> None:
        if w in self.waiters:
            self.waiters.remove(w)

    def may_go(self, w: Waiter, hit: bool = False) -> bool:
        """Whether `w` may take a free slot now; a granted overtake is charged to the oldest waiter."""
        head = self.waiters[0] if self.waiters else w
        if w is head:
            return True
        if hit and head.skipped < self.overtake_max:
            head.skipped += 1
            return True
        return False

    def may_go_shared(self, w: Waiter, free_pages: int, active_adapter: int | None = None,
                      *, charge_bypass: bool = True, full_owners: frozenset = frozenset()) -> bool:
        """Grant the oldest fitting waiter, with one bypass of the blocked head."""
        def fits(waiter: Waiter) -> bool:
            needed = waiter.minimum_pages if full_owners - waiter.previous_owners else waiter.pages
            return needed <= free_pages and (active_adapter is None or waiter.adapter == active_adapter)

        head = self.waiters[0]
        if fits(head):
            return w is head
        if head.skipped:
            return False
        for waiter in itertools.islice(self.waiters, 1, None):
            if fits(waiter):
                if waiter is not w:
                    return False
                head.skipped += int(charge_bypass)
                return True
        return False

    def __len__(self) -> int:
        return len(self.waiters)
