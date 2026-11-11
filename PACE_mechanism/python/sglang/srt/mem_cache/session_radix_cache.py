# PACE 会话持有者标记及释放，保留原有节点生命周期逻辑。
# 依赖与归档相兼容的 SGLang；不是独立完整推理引擎。
from __future__ import annotations
import logging
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
logger = logging.getLogger(__name__)
_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192

class SessionRadixCacheMixin:

    def _reset_session_radix_state(self) -> None:
        self._session_leaves = defaultdict(set)
        self._closed_session_ids = OrderedDict()

    def _ensure_session_radix_state(self) -> None:
        if not hasattr(self, '_session_leaves'):
            self._reset_session_radix_state()

    def set_queued_sessions(self, sids) -> None:
        strat = getattr(self, 'eviction_strategy', None)
        tracker = getattr(strat, 'tracker', None)
        if tracker is not None and hasattr(tracker, 'set_queued'):
            tracker.set_queued(sids)
            return
        if isinstance(sids, dict):
            self._queued_sids = set(sids.keys())
        elif sids:
            self._queued_sids = set(sids)
        else:
            self._queued_sids = set()

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def _discard_session_leaf(self, node) -> None:
        session_ids = getattr(node, 'session_ids', None)
        if not session_ids or not hasattr(self, '_session_leaves'):
            return
        for sid in tuple(session_ids):
            leaves = self._session_leaves.get(sid)
            if leaves is not None:
                leaves.discard(node)
                if not leaves and sid not in self._closed_session_ids:
                    self._session_leaves.pop(sid, None)
                    tracker = getattr(self, '_pace_tracker', None)
                    if tracker is not None:
                        tracker.pop(sid)
        if hasattr(node, 'session_ids'):
            delattr(node, 'session_ids')
        sync = getattr(self, '_ruei_sync_node', None)
        if sync is not None:
            sync(node)

    def _tag_session_leaf(self, req: Req, radix_key, node=None) -> None:
        if not getattr(self, 'enable_session_radix_cache', False):
            return
        self._ensure_session_radix_state()
        sid = getattr(req, 'session_id', None)
        if sid is None or sid in self._closed_session_ids:
            return
        if node is None:
            node = self.match_prefix(MatchPrefixParams(key=radix_key)).last_device_node
        if node is not None and node is not self.root_node:
            session_ids = getattr(node, 'session_ids', None)
            if session_ids is None:
                session_ids = set()
                node.session_ids = session_ids
            session_ids.add(sid)
            self._session_leaves[sid].add(node)
            sync = getattr(self, '_ruei_sync_node', None)
            if sync is not None:
                sync(node)

    def release_radix_session(self, session_id: str) -> int:
        self._ensure_session_radix_state()
        self._remember_closed_session(session_id)
        indexed = self._session_leaves.pop(session_id, set())
        tracker = getattr(self, '_pace_tracker', None)
        if tracker is not None:
            tracker.pop(session_id)
        freed = 0
        for leaf in indexed:
            if session_id not in getattr(leaf, 'session_ids', set()):
                continue
            node = leaf
            while True:
                session_ids = getattr(node, 'session_ids', None)
                if session_ids is not None:
                    session_ids.discard(session_id)
                    if not session_ids:
                        delattr(node, 'session_ids')
                sync = getattr(self, '_ruei_sync_node', None)
                if sync is not None:
                    sync(node)
                if node is self.root_node or node.lock_ref != 0 or len(node.children) != 0 or (node not in self.evictable_leaves) or getattr(node, 'session_ids', None):
                    break
                parent = node.parent
                self.token_to_kv_pool_allocator.free(node.value)
                self._delete_leaf(node)
                freed += 1
                node = parent
        return freed
