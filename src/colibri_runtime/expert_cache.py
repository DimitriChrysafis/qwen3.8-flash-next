from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    pinned: int = 0


class ExpertCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._items: OrderedDict[str, tuple[Any, bool, float]] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        if key not in self._items:
            self.stats.misses += 1
            return None
        value, pinned, _ = self._items.pop(key)
        self._items[key] = (value, pinned, monotonic())
        self.stats.hits += 1
        return value

    def put(self, key: str, value: Any, pin: bool = False) -> None:
        if key in self._items:
            _, old_pin, _ = self._items.pop(key)
            pin = pin or old_pin
        self._items[key] = (value, pin, monotonic())
        self._evict_if_needed()

    def pin(self, key: str) -> bool:
        if key not in self._items:
            return False
        value, _, ts = self._items.pop(key)
        self._items[key] = (value, True, ts)
        return True

    def unpin(self, key: str) -> bool:
        if key not in self._items:
            return False
        value, _, ts = self._items.pop(key)
        self._items[key] = (value, False, ts)
        return True

    def _evict_if_needed(self) -> None:
        while len(self._items) > self.max_items:
            victim = None
            for k, (_, pinned, _) in self._items.items():
                if not pinned:
                    victim = k
                    break
            if victim is None:
                break
            self._items.pop(victim)
            self.stats.evictions += 1
        self.stats.pinned = sum(1 for _, p, _ in self._items.values() if p)

    def snapshot(self) -> dict[str, int]:
        return {
            "size": len(self._items),
            "max_items": self.max_items,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "pinned": self.stats.pinned,
        }
