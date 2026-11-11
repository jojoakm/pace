# PACE 已测 RUEI 维护补丁：事件计账、轮内快照和档位变化传播。
# 依赖与归档相兼容的 SGLang；不是独立完整推理引擎。
import math
import os
from collections import defaultdict

def install(Tracker, Quota, Index):
    if os.environ.get('PACE_RUEI_INCREMENTAL', '0') != '1':
        return
    originals = {name: getattr(Index, name) for name in ['__init__', 'reset', 'sync_node', '_refresh_session', '_refresh_online_ttl', '_refresh_quota_occupancy']}
    begin_original = Quota.begin_round

    def clear(index):
        index._inc_nodes = {}
        index._inc_totals = defaultdict(dict)
        index._inc_pending = set()
        index._inc_occ_changed = set()
        index._inc_n0 = set()
        index.strategy._occ = {}
        index.strategy._W = 0.0

    def init(index, cache, strategy, now):
        originals['__init__'](index, cache, strategy, now)
        assert getattr(strategy, 'is_quota', False), '增量占用仅验证Quota路径'
        assert os.environ.get('PACE_BATCH_EVICT', '0') != '1'
        strategy._incremental_index = index
        clear(index)
        for node in cache.evictable_leaves:
            account(index, node)

    def reset(index, now):
        originals['reset'](index, now)
        clear(index)

    def account(index, node):
        previous = index._inc_nodes.get(node)
        owners = frozenset(getattr(node, 'session_ids', ()) or ())
        current = (owners, len(node.value)) if owners and node in index.cache.evictable_leaves else None
        if current == previous:
            return
        for (spec, sign) in [(previous, -1), (current, 1)]:
            if spec is None:
                continue
            (sids, length) = spec
            divisor = len(sids)
            for sid in sids:
                parts = index._inc_totals[sid]
                value = parts.get(divisor, 0) + sign * length
                if value:
                    parts[divisor] = value
                else:
                    parts.pop(divisor, None)
                if not parts:
                    index._inc_totals.pop(sid, None)
                index._inc_pending.add(sid)
        if current is None:
            index._inc_nodes.pop(node, None)
        else:
            index._inc_nodes[node] = current

    def sync_node(index, node, now=None):
        account(index, node)
        return originals['sync_node'](index, node, now)

    def begin_round(strategy, leaves):
        index = getattr(strategy, '_incremental_index', None)
        if index is None:
            return begin_original(strategy, leaves)
        for sid in index._inc_pending:
            parts = index._inc_totals.get(sid)
            previous = strategy._occ.get(sid)
            current = math.fsum((length / divisor for (divisor, length) in parts.items())) if parts else None
            if current is None:
                strategy._occ.pop(sid, None)
            else:
                strategy._occ[sid] = current
            if previous != current:
                index._inc_occ_changed.add(sid)
        index._inc_pending.clear()
        strategy._W = 0.0

    def refresh_occupancy(index, now):
        affected = set()
        for sid in index._inc_occ_changed:
            affected.update(index.cache._session_leaves.get(sid, ()))
        index._inc_occ_changed.clear()
        index._quota_occ_seen = index.strategy._occ
        for node in affected:
            index.sync_node(node, now)

    def refresh_session(index, sid, now, sync_nodes=True):
        previous = index.session_rate.get(sid)
        existed = sid in index.session_rate
        originals['_refresh_session'](index, sid, now, sync_nodes=False)
        state = index.tracker._state.get(sid)
        if state is not None and state.get('n', 0) == 0:
            index._inc_n0.add(sid)
        else:
            index._inc_n0.discard(sid)
        changed = existed != (sid in index.session_rate) or previous != index.session_rate.get(sid)
        if index.initialized and sync_nodes and changed:
            for node in tuple(index.cache._session_leaves.get(sid, ())):
                index.sync_node(node, now)

    def refresh_ttl(index, now):
        epoch = getattr(index.tracker, '_ruei_n0_epoch', 0)
        if epoch == index._n0_epoch_seen:
            return
        index._n0_epoch_seen = epoch
        if not getattr(index.tracker, 'n0_online', ''):
            return
        ttl = index.tracker._eff_ttl()
        if ttl == index._n0_ttl_seen:
            return
        index._n0_ttl_seen = ttl
        for sid in tuple(index._inc_n0):
            state = index.tracker._state.get(sid)
            if state is None or state.get('n', 0) != 0:
                index._inc_n0.discard(sid)
                continue
            if index.lazy_ttl and (not index.cache._session_leaves.get(sid)):
                index._ttl_stale.add(sid)
            else:
                index._refresh_session(sid, now)
    Index.__init__ = init
    Index.reset = reset
    Index.sync_node = sync_node
    Index._refresh_session = refresh_session
    Index._refresh_online_ttl = refresh_ttl
    Index._refresh_quota_occupancy = refresh_occupancy
    Quota.begin_round = begin_round
