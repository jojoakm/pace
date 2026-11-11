# PACE RUEI：几何档位、解析事件、时间轮和版本堆。
# 依赖与归档相兼容的 SGLang；不是独立完整推理引擎。
from __future__ import annotations
import heapq
import json
import logging
import math
import os
import time
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from sglang.srt.mem_cache.evict_policy import PaceAwareStrategy
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
logger = logging.getLogger(__name__)

class SessionDeadlineWheel:

    def __init__(self, now: float, resolution: float, size: int):
        self.resolution = max(0.001, resolution)
        self.size = max(64, size)
        self.slots: list[list[tuple[int, str, int]]] = [[] for _ in range(self.size)]
        self.current_tick = int(math.floor(now / self.resolution + 1e-09))
        self.scheduled = 0
        self.fired = 0

    def clear(self, now: float) -> None:
        self.slots = [[] for _ in range(self.size)]
        self.current_tick = int(math.floor(now / self.resolution + 1e-09))
        self.scheduled = 0
        self.fired = 0

    def schedule(self, sid: str, deadline: Optional[float], version: int) -> None:
        if deadline is None or not math.isfinite(deadline):
            return
        target_tick = max(self.current_tick + 1, int(math.ceil(deadline / self.resolution - 1e-09)))
        self.slots[target_tick % self.size].append((target_tick, sid, version))
        self.scheduled += 1

    def advance(self, now: float) -> list[tuple[str, int]]:
        target = int(math.floor(now / self.resolution + 1e-09))
        if target <= self.current_tick:
            return []
        due: list[tuple[str, int]] = []
        steps = target - self.current_tick
        if steps > self.size * 2:
            for (i, slot) in enumerate(self.slots):
                keep = []
                for (deadline_tick, sid, version) in slot:
                    if deadline_tick <= target:
                        due.append((sid, version))
                    else:
                        keep.append((deadline_tick, sid, version))
                self.slots[i] = keep
            self.current_tick = target
        else:
            while self.current_tick < target:
                self.current_tick += 1
                idx = self.current_tick % self.size
                (slot, self.slots[idx]) = (self.slots[idx], [])
                for (deadline_tick, sid, version) in slot:
                    if deadline_tick <= self.current_tick:
                        due.append((sid, version))
                    else:
                        self.slots[deadline_tick % self.size].append((deadline_tick, sid, version))
        self.fired += len(due)
        return due

class RhythmUtilityEvictionIndex:

    def __init__(self, cache: 'RadixCache', strategy: 'PaceAwareStrategy', now: float):
        self.cache = cache
        self.strategy = strategy
        self.tracker = strategy.tracker
        self.ratio = float(os.environ.get('PACE_RUEI_RATIO', '2.0'))
        if self.ratio <= 1.0:
            raise ValueError('PACE_RUEI_RATIO must be > 1')
        resolution = float(os.environ.get('PACE_RUEI_TICK', '0.1'))
        wheel_size = int(os.environ.get('PACE_RUEI_WHEEL_SIZE', '4096'))
        self.wheel = SessionDeadlineWheel(now, resolution, wheel_size)
        self.heap: list[tuple[float, int, int, 'TreeNode']] = []
        self.session_rate: dict[str, float] = {}
        self.session_version: dict[str, int] = {}
        self.initialized = False
        self.pops = 0
        self.stale_pops = 0
        self.node_updates = 0
        self.session_updates = 0
        self.compactions = 0
        self.prepare_calls = 0
        self._quota_occ_seen = None
        self._n0_epoch_seen = -1
        self._n0_ttl_seen = None
        self.lazy_ttl = os.environ.get('PACE_RUEI_LAZY_TTL', '0') == '1'
        self._ttl_stale: set = set()
        unsupported = []
        if self.tracker.imminence:
            unsupported.append('PACE_IMMINENCE')
        if getattr(strategy, 'value_floor', 0) > 0:
            unsupported.append('PACE_VALUE_FLOOR')
        if getattr(strategy, 'is_quota', False):
            if getattr(strategy, 'budget_floor_frac', 0) > 0:
                unsupported.append('PACE_BUDGET_FLOOR')
            if getattr(strategy, 'ignore_occupancy', False):
                unsupported.append('PACE_QUOTA_IGNORE_OCC')
            if getattr(self.tracker, 'owner_discount', 0) > 0:
                unsupported.append('PACE_OWNER_DISCOUNT')
            if getattr(self.tracker, 'hard_quota', False):
                unsupported.append('PACE_HARD_QUOTA')
            if os.environ.get('PACE_DEAD_FILTER', '0') == '1':
                unsupported.append('PACE_DEAD_FILTER')
            if getattr(strategy, 'head_preserve', False):
                unsupported.append('PACE_HEAD_PRESERVE')
        if getattr(self.tracker, 'k_online', False) or getattr(self.tracker, 'shrink', False):
            unsupported.append('global adaptive K or shrinkage')
        if getattr(self.tracker, 'turns_boost', False) or getattr(self.tracker, 'tau_cv', False) or getattr(self.tracker, 'lambda_floor', 0) > 0:
            unsupported.append('unvalidated exploratory rate modifiers')
        if getattr(self.tracker, 'n0_mode', '') not in ('', 'cap', 'ttl_only'):
            unsupported.append('unvalidated cold-start mode')
        if getattr(self.tracker, 'cold_start', False) and getattr(self.tracker, 'n0_mode', ''):
            unsupported.append('combined cold-start flags')
        if unsupported:
            raise ValueError('RUEI prototype does not support: ' + ', '.join(unsupported))
        self.tracker._ruei_dirty = set(self.tracker._state)

    def reset(self, now: float) -> None:
        self.heap.clear()
        self.session_rate.clear()
        self.session_version.clear()
        self.wheel.clear(now)
        self.initialized = False
        self._quota_occ_seen = None
        self._n0_epoch_seen = -1
        self._n0_ttl_seen = None
        self._ttl_stale.clear()

    def _refresh_session(self, sid: str, now: float, sync_nodes: bool=True) -> None:
        self._ttl_stale.discard(sid)
        version = self.session_version.get(sid, 0) + 1
        self.session_version[sid] = version
        (rate, deadline) = self.tracker.ruei_rate_and_deadline(sid, now, self.ratio)
        previous = self.session_rate.get(sid)
        if rate is None:
            self.session_rate.pop(sid, None)
        else:
            self.session_rate[sid] = rate
            self.wheel.schedule(sid, deadline, version)
        self.session_updates += 1
        if self.initialized and sync_nodes:
            for node in tuple(self.cache._session_leaves.get(sid, ())):
                self.sync_node(node, now)

    def initialize(self, now: float) -> None:
        for sid in tuple(self.tracker._state):
            self._refresh_session(sid, now)
        self.heap.clear()
        self.initialized = True
        for node in tuple(self.cache.evictable_leaves):
            self.sync_node(node, now)
        self.tracker._ruei_dirty.clear()

    def _refresh_quota_occupancy(self, now: float) -> None:
        if not getattr(self.strategy, 'is_quota', False):
            return
        current = self.strategy._occ
        if current is self._quota_occ_seen:
            return
        previous = self._quota_occ_seen or {}
        changed = {sid for sid in previous.keys() | current.keys() if previous.get(sid) != current.get(sid)}
        self._quota_occ_seen = current
        affected = set()
        for sid in changed:
            affected.update(self.cache._session_leaves.get(sid, ()))
        for node in affected:
            self.sync_node(node, now)

    def _refresh_online_ttl(self, now: float) -> None:
        epoch = getattr(self.tracker, '_ruei_n0_epoch', 0)
        if epoch == self._n0_epoch_seen:
            return
        self._n0_epoch_seen = epoch
        if not getattr(self.tracker, 'n0_online', ''):
            return
        ttl = self.tracker._eff_ttl()
        if ttl == self._n0_ttl_seen:
            return
        self._n0_ttl_seen = ttl
        lazy = self.lazy_ttl
        indexed = self.cache._session_leaves
        for (sid, state) in tuple(self.tracker._state.items()):
            if state.get('n', 0) == 0:
                if lazy and (not indexed.get(sid)):
                    self._ttl_stale.add(sid)
                    continue
                self._refresh_session(sid, now)

    def prepare(self, now: float) -> None:
        self.prepare_calls += 1
        if not self.initialized:
            self.initialize(now)
            self._quota_occ_seen = getattr(self.strategy, '_occ', None)
            self._n0_epoch_seen = getattr(self.tracker, '_ruei_n0_epoch', 0)
            self._n0_ttl_seen = self.tracker._eff_ttl() if getattr(self.tracker, 'n0_online', '') else None
            return
        dirty = tuple(self.tracker._ruei_dirty)
        self.tracker._ruei_dirty.clear()
        for sid in dirty:
            self._refresh_session(sid, now)
        for (sid, version) in self.wheel.advance(now):
            if self.session_version.get(sid) == version:
                self._refresh_session(sid, now)
        self._refresh_online_ttl(now)
        self._refresh_quota_occupancy(now)
        live = len(self.cache.evictable_leaves)
        if len(self.heap) > max(1024, live * 4):
            self._rebuild_nodes(now)
        if self.prepare_calls % 500 == 0:
            pass

    def _approx_priority(self, node: 'TreeNode', now: float) -> float:
        if getattr(self.strategy, 'is_quota', False):
            owners = getattr(node, 'session_ids', None)
            occ = getattr(self.strategy, '_occ', None)
            worst = None
            if owners and occ:
                for sid in owners:
                    o = occ.get(sid)
                    if o is None:
                        continue
                    if sid in self._ttl_stale:
                        self._refresh_session(sid, now, sync_nodes=False)
                    q = self.session_rate.get(sid)
                    if q is None or q <= 0.0:
                        return (float('-inf'), node.last_access_time)
                    ratio = o / q
                    if worst is None or ratio > worst:
                        worst = ratio
            if worst is not None:
                return (-worst, node.last_access_time)
            return self.strategy.get_priority(node)
        owners = getattr(node, 'session_ids', None)
        rate = 0.0
        if owners:
            for sid in owners:
                if sid in self._ttl_stale:
                    self._refresh_session(sid, now, sync_nodes=False)
                q = self.session_rate.get(sid)
                if q is not None:
                    rate = max(rate, q) if self.strategy.owner_max else rate + q
        if rate > 0.0:
            value = 1.0 if self.strategy.no_value else self.strategy._value(node)
            return value * rate
        return self.strategy.get_priority(node)

    def sync_node(self, node: 'TreeNode', now: Optional[float]=None) -> None:
        if not self.initialized:
            return
        version = getattr(node, '_ruei_version', 0) + 1
        node._ruei_version = version
        if node is self.cache.root_node or node not in self.cache.evictable_leaves:
            return
        if now is None:
            now = time.perf_counter()
        priority = self._approx_priority(node, now)
        heapq.heappush(self.heap, (priority, node.id, version, node))
        self.node_updates += 1

    def _rebuild_nodes(self, now: float) -> None:
        self.heap.clear()
        self.compactions += 1
        for node in tuple(self.cache.evictable_leaves):
            self.sync_node(node, now)

    def pop(self, now: float) -> Optional['TreeNode']:
        self.prepare(now)
        rebuilt = False
        while True:
            while self.heap:
                (_, _, version, node) = heapq.heappop(self.heap)
                if version != getattr(node, '_ruei_version', -1) or node not in self.cache.evictable_leaves or node.lock_ref != 0:
                    self.stale_pops += 1
                    continue
                self.pops += 1
                return node
            if rebuilt or not self.cache.evictable_leaves:
                return None
            self._rebuild_nodes(now)
            rebuilt = True
