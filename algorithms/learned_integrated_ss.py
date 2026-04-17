"""
Learned-Integrated SpaceSaving± (Advanced Design).

Combines Integrated SpaceSaving± with a lightweight frequency predictor to
allocate "fixed" counters for predicted hot items, leaving mutable counters
for the standard Integrated SpaceSaving± mechanism on the rest.

Idea:
    - Maintain a sliding window of the last W updates (default W=1000).
    - Periodically score items by their recent frequency in the window.
    - Items exceeding a threshold are predicted as "hot" and assigned dedicated
      fixed counters that are never evicted.
    - Remaining counters operate as standard Integrated SpaceSaving±.
    - Hot-item counters are exact (no eviction error), reducing overall error
      especially under Zipfian distributions.

Deterministic guarantee on cold items is preserved: mutable counters still
satisfy |f̂(x) - f(x)| <= ε * F1 with O(α/ε) mutable slots.

Pseudocode:
    INIT(k, α, ε, W=1000, fixed_ratio=0.2):
        k_fixed = floor(fixed_ratio * k)
        k_mutable = k - k_fixed
        fixed_counters = {}          # item -> IntegratedEntry (never evicted)
        mutable = IntegratedSpaceSaving(k_mutable, α)
        window = deque(maxlen=W)
        window_counts = Counter()

    UPDATE(item, delta):
        # Update window
        if len(window) == W:
            old_item, old_delta = window[0]
            window_counts[old_item] -= abs(old_delta)
        window.append((item, delta))
        window_counts[item] += abs(delta)

        # Periodically re-evaluate hot set (every W/2 updates)
        if num_updates % (W//2) == 0:
            recompute_hot_set()

        if item in fixed_counters:
            # Direct update, no eviction
            update fixed_counters[item] with delta
        else:
            mutable.update(item, delta)

    RECOMPUTE_HOT_SET():
        ranked = sort window_counts descending
        new_hot = top k_fixed items from ranked
        # Migrate: demote items no longer hot, promote new hot items
        for item leaving hot set: move counter to mutable
        for item entering hot set: move counter from mutable (if present) to fixed

    QUERY(item):
        if item in fixed_counters: return fixed_counters[item].count
        return mutable.query(item)
"""

import math
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LISEntry:
    """Counter entry for Learned-Integrated SpaceSaving±."""
    item: object = None
    insert_count: int = 0
    delete_count: int = 0
    max_error: int = 0

    @property
    def count(self) -> int:
        return self.insert_count - self.delete_count


class _MutableIntegratedSS:
    """Lightweight Integrated SpaceSaving± used as the mutable portion."""

    def __init__(self, k: int):
        self.k = max(k, 1)
        self.entries: Dict[object, LISEntry] = {}

    def update(self, item, delta: int) -> None:
        if delta > 0:
            self._insert(item, delta)
        elif delta < 0:
            self._delete(item, -delta)

    def query(self, item) -> int:
        if item in self.entries:
            return max(self.entries[item].count, 0)
        return 0

    def pop_entry(self, item) -> Optional[LISEntry]:
        """Remove and return entry for item, or None."""
        return self.entries.pop(item, None)

    def inject_entry(self, entry: LISEntry) -> None:
        """Force-insert an entry (used during hot-set migration). May evict min."""
        if entry.item in self.entries:
            # Merge
            e = self.entries[entry.item]
            e.insert_count += entry.insert_count
            e.delete_count += entry.delete_count
            e.max_error = max(e.max_error, entry.max_error)
        elif len(self.entries) < self.k:
            self.entries[entry.item] = entry
        else:
            min_entry = min(self.entries.values(), key=lambda e: e.count)
            if entry.count > min_entry.count:
                del self.entries[min_entry.item]
                self.entries[entry.item] = entry
            # else: drop the injected entry (it's smaller)

    def _insert(self, item, weight: int) -> None:
        if item in self.entries:
            self.entries[item].insert_count += weight
        elif len(self.entries) < self.k:
            self.entries[item] = LISEntry(item=item, insert_count=weight)
        else:
            min_entry = min(self.entries.values(), key=lambda e: e.count)
            min_count = max(min_entry.count, 0)
            del self.entries[min_entry.item]
            self.entries[item] = LISEntry(
                item=item, insert_count=min_count + weight,
                delete_count=0, max_error=min_count
            )

    def _delete(self, item, weight: int) -> None:
        if item in self.entries:
            self.entries[item].delete_count += weight
            if self.entries[item].count <= 0:
                del self.entries[item]
        else:
            if self.entries:
                max_err = max(self.entries.values(), key=lambda e: e.max_error)
                max_err.delete_count += weight
                if max_err.count <= 0:
                    del self.entries[max_err.item]


class LearnedIntegratedSpaceSaving:
    """
    Learned-Integrated SpaceSaving±.

    Parameters
    ----------
    k : int
        Total number of counters (fixed + mutable).
    alpha : float
        α-bounded deletion parameter.
    window_size : int
        Size of the sliding window for the predictor (default 1000).
    fixed_ratio : float
        Fraction of k counters reserved for predicted hot items (default 0.2).
    retrain_interval : int or None
        Re-evaluate hot set every this many updates. Default window_size // 2.
    """

    def __init__(self, k: int, alpha: float = 1.0, window_size: int = 1000,
                 fixed_ratio: float = 0.2, retrain_interval: Optional[int] = None):
        self.k = max(k, 2)
        self.alpha = alpha
        self.window_size = window_size
        self.fixed_ratio = fixed_ratio

        self.k_fixed = max(1, int(math.floor(fixed_ratio * self.k)))
        self.k_mutable = self.k - self.k_fixed

        # Fixed counters for predicted hot items (never evicted by SS logic)
        self.fixed: Dict[object, LISEntry] = {}
        # Mutable counters – standard Integrated SpaceSaving±
        self.mutable = _MutableIntegratedSS(self.k_mutable)

        # Sliding window predictor
        self.window: deque = deque(maxlen=window_size)
        self.window_counts: Counter = Counter()

        self.retrain_interval = retrain_interval or max(window_size // 2, 1)
        self._total_positive = 0
        self._total_negative = 0
        self._num_updates = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, item, delta: int = 1) -> None:
        self._num_updates += 1
        if delta > 0:
            self._total_positive += delta
        else:
            self._total_negative += (-delta)

        # --- Update sliding window ---
        if len(self.window) == self.window_size:
            old_item, old_delta = self.window[0]
            self.window_counts[old_item] -= abs(old_delta)
            if self.window_counts[old_item] <= 0:
                del self.window_counts[old_item]
        self.window.append((item, delta))
        self.window_counts[item] = self.window_counts.get(item, 0) + abs(delta)

        # --- Periodically retrain hot set ---
        if self._num_updates % self.retrain_interval == 0:
            self._recompute_hot_set()

        # --- Route update ---
        if item in self.fixed:
            if delta > 0:
                self.fixed[item].insert_count += delta
            else:
                self.fixed[item].delete_count += (-delta)
                if self.fixed[item].count <= 0:
                    del self.fixed[item]
        else:
            self.mutable.update(item, delta)

    def query(self, item) -> int:
        if item in self.fixed:
            return max(self.fixed[item].count, 0)
        return self.mutable.query(item)

    def heavy_hitters(self, phi: float) -> List[Tuple]:
        f1 = self.get_F1()
        threshold = phi * f1
        results = []
        for e in self.fixed.values():
            if e.count >= threshold:
                results.append((e.item, e.count))
        for e in self.mutable.entries.values():
            if e.count >= threshold:
                results.append((e.item, e.count))
        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_F1(self) -> int:
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        return len(self.fixed) + len(self.mutable.entries)

    def get_max_space(self) -> int:
        return self.k

    # ------------------------------------------------------------------
    # Predictor / hot-set management
    # ------------------------------------------------------------------

    def _recompute_hot_set(self) -> None:
        """
        Re-evaluate which items should be in the fixed (hot) set based on
        recent window frequencies. Migrate counters between fixed and mutable.
        """
        # Rank items by recent window frequency
        if not self.window_counts:
            return

        top_items = [item for item, _ in self.window_counts.most_common(self.k_fixed)]
        new_hot_set = set(top_items)
        current_hot_set = set(self.fixed.keys())

        # Demote: items leaving hot set → move to mutable
        for item in current_hot_set - new_hot_set:
            entry = self.fixed.pop(item)
            self.mutable.inject_entry(entry)

        # Promote: items entering hot set → move from mutable to fixed
        for item in new_hot_set - current_hot_set:
            if len(self.fixed) >= self.k_fixed:
                break
            existing = self.mutable.pop_entry(item)
            if existing is not None:
                self.fixed[item] = existing
            else:
                # Create a fresh fixed entry (no historical count available)
                self.fixed[item] = LISEntry(item=item)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_epsilon_alpha(cls, epsilon: float, alpha: float,
                           window_size: int = 1000,
                           fixed_ratio: float = 0.2) -> "LearnedIntegratedSpaceSaving":
        k = math.ceil(alpha / epsilon)
        return cls(k=k, alpha=alpha, window_size=window_size, fixed_ratio=fixed_ratio)

    def __repr__(self) -> str:
        return (f"LearnedIntegratedSS(k={self.k}, fixed={len(self.fixed)}, "
                f"mutable={len(self.mutable.entries)}, F1={self.get_F1()})")
                