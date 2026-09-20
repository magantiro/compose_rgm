"""Bounded, process-local reuse of deterministic proposal work, never oracle labels.

Entries are scoped to one running implementation. They are deliberately not
restored from disk: snapshots retain attempts/RNG, and a resumed worker starts
with a cold cache. No unqueried candidate is suppressed and no draw is skipped.
"""

from collections import Counter, OrderedDict


class ProgramWorkCache:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("proposal cache capacity must be a nonnegative integer")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits, self.misses = Counter(), Counter()

    def get(self, namespace, key, compute):
        if not self.capacity:
            return compute()
        address = (namespace, key)
        if address in self.entries:
            self.hits[namespace] += 1
            return self.entries[address]
        self.misses[namespace] += 1
        value = compute()
        self.entries[address] = value
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value

    def report(self):
        return {
            "capacity": self.capacity,
            "entries": len(self.entries),
            "hits": dict(self.hits),
            "misses": dict(self.misses),
            "persistence": "process_local_cold_on_resume",
        }
