from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any, Hashable


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    pinned: int = 0
    bytes: int = 0
    peak_bytes: int = 0
    rejected: int = 0


@dataclass
class _Entry:
    value: Any
    nbytes: int
    pinned: bool
    hot: bool
    frequency: int
    touched: float


class ExpertCache:
    def __init__(
        self,
        max_items: int | None = None,
        max_bytes: int | None = None,
        policy: str = "adaptive",
    ) -> None:
        if max_items is None and max_bytes is None:
            raise ValueError("max_items or max_bytes is required")
        if policy not in {"lru", "lfu", "adaptive"}:
            raise ValueError("policy must be lru, lfu, or adaptive")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.policy = policy
        self._items: OrderedDict[Hashable, _Entry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def get(self, key: Hashable, count_miss: bool = True) -> Any | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                if count_miss:
                    self.stats.misses += 1
                return None
            self._items.move_to_end(key)
            entry.frequency += 1
            entry.touched = monotonic()
            self.stats.hits += 1
            return entry.value

    def peek(self, key: Hashable) -> Any | None:
        with self._lock:
            entry = self._items.get(key)
            return None if entry is None else entry.value

    def put(
        self,
        key: Hashable,
        value: Any,
        nbytes: int | None = None,
        pin: bool = False,
        hot: bool = False,
    ) -> bool:
        size = int(nbytes if nbytes is not None else getattr(value, "nbytes", 1))
        if size < 0:
            raise ValueError("nbytes must be nonnegative")
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.stats.bytes -= old.nbytes
                pin = pin or old.pinned
                hot = hot or old.hot
            if self.max_bytes is not None and size > self.max_bytes:
                self.stats.rejected += 1
                return False
            while ((self.max_items is not None and len(self._items) + 1 > self.max_items) or
                   (self.max_bytes is not None and self.stats.bytes + size > self.max_bytes)):
                victim = self._victim()
                if victim is None:
                    self.stats.rejected += 1
                    self._update_pinned()
                    return False
                entry = self._items.pop(victim)
                self.stats.bytes -= entry.nbytes
                self.stats.evictions += 1
            self._items[key] = _Entry(value, size, pin, hot, 1, monotonic())
            self.stats.bytes += size
            self.stats.peak_bytes = max(self.stats.peak_bytes, self.stats.bytes)
            self._update_pinned()
            return True

    def pin(self, key: Hashable) -> bool:
        with self._lock:
            if key not in self._items:
                return False
            self._items[key].pinned = True
            self._update_pinned()
            return True

    def unpin(self, key: Hashable) -> bool:
        with self._lock:
            if key not in self._items:
                return False
            self._items[key].pinned = False
            self._update_pinned()
            self._evict_if_needed()
            return True

    def mark_hot(self, key: Hashable, hot: bool = True) -> bool:
        with self._lock:
            if key not in self._items:
                return False
            self._items[key].hot = hot
            return True

    def _over(self) -> bool:
        return ((self.max_items is not None and len(self._items) > self.max_items) or
                (self.max_bytes is not None and self.stats.bytes > self.max_bytes))

    def _victim(self, protected: Hashable | None = None) -> Hashable | None:
        candidates = [(k, e) for k, e in self._items.items() if not e.pinned and k != protected]
        if not candidates:
            if protected is not None:
                e = self._items.get(protected)
                return protected if e is not None and not e.pinned else None
            return None
        cold = [(k, e) for k, e in candidates if not e.hot]
        candidates = cold or candidates
        if self.policy == "lru":
            return min(candidates, key=lambda item: item[1].touched)[0]
        if self.policy == "lfu":
            return min(candidates, key=lambda item: (item[1].frequency, item[1].touched))[0]
        now = monotonic()
        return min(
            candidates,
            key=lambda item: ((item[1].frequency + (4 if item[1].hot else 0)) /
                              max(item[1].nbytes, 1), item[1].touched - now),
        )[0]

    def _evict_if_needed(self, protected: Hashable | None = None) -> None:
        while self._over():
            victim = self._victim(protected)
            if victim is None:
                break
            entry = self._items.pop(victim)
            self.stats.bytes -= entry.nbytes
            self.stats.evictions += 1
        self._update_pinned()

    def _update_pinned(self) -> None:
        self.stats.pinned = sum(entry.pinned for entry in self._items.values())

    def snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            out = asdict(self.stats)
            out.update(size=len(self._items), hot=sum(entry.hot for entry in self._items.values()),
                       max_items=self.max_items, max_bytes=self.max_bytes, policy=self.policy)
            return out
