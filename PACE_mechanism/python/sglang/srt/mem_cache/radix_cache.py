# PACE 的 SGLang 基数缓存接入；已移除其他基线与诊断分支。
# 依赖与归档相兼容的 SGLang；不是独立完整推理引擎。
from __future__ import annotations
from sglang.srt.mem_cache.evict_policy import LRUStrategy, get_eviction_strategy
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
'\nCopyright 2023-2024 SGLang Team\nLicensed under the Apache License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an "AS IS" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n'
import heapq
import json
import logging
import os
import sys
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union
import torch
logger = logging.getLogger(__name__)
import math
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, DecLockRefParams, DecLockRefResult, EvictParams, EvictResult, IncLockRefResult, InsertParams, InsertResult, MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.evict_policy import PaceAwareStrategy, SessionPaceTracker
from sglang.srt.mem_cache.ruei_index import RhythmUtilityEvictionIndex
from sglang.srt.mem_cache.session_radix_cache import SessionRadixCacheMixin
from sglang.srt.mem_cache.utils import get_hash_str, split_node_hash_value
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

class RadixKey:
    __slots__ = ('token_ids', 'extra_key', 'is_bigram', 'limit')

    def __init__(self, token_ids: array[int], extra_key: Optional[str]=None, is_bigram: bool=False, limit: Optional[int]=None):
        self.token_ids = token_ids
        self.extra_key = extra_key
        self.is_bigram = is_bigram
        self.limit = limit

    def _raw_len(self) -> int:
        n = len(self.token_ids)
        if self.limit is not None and self.limit < n:
            return self.limit
        return n

    def raw_token_ids(self) -> array:
        n = self._raw_len()
        t = self.token_ids
        return t if n == len(t) else t[:n]

    def __len__(self) -> int:
        n = self._raw_len()
        if self.is_bigram:
            return n - 1 if n > 0 else 0
        return n

    def __iter__(self) -> Iterator:
        t = self.token_ids
        n = self._raw_len()
        if self.is_bigram:
            for i in range(n - 1 if n > 0 else 0):
                yield (t[i], t[i + 1])
        elif n == len(t):
            yield from t
        else:
            for i in range(n):
                yield t[i]

    def __getitem__(self, idx: Union[int, slice]) -> RadixKey:
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f'RadixKey index out of range: {idx}')
            idx = slice(idx, idx + 1)
        (start, stop, step) = idx.indices(len(self))
        if step != 1:
            raise ValueError('RadixKey slice step must be 1')
        if self.is_bigram:
            raw = self.token_ids[start:stop + 1] if stop > start else array('q')
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{('...' if len(self.token_ids) > 10 else '')}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> RadixKey:
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(self, is_eagle: bool, value: Optional[torch.Tensor]=None) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        if is_eagle and (not self.is_bigram):
            self.is_bigram = True
            if value is not None:
                value = value[:len(self)]
        return (self, value)

    def _check_compatible(self, other: RadixKey) -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(f'RadixKey operations require matching extra_key, but got self.extra_key={self.extra_key!r} != other.extra_key={other.extra_key!r}')

    def match(self, other: RadixKey, page_size: int=1) -> int:
        self._check_compatible(other)
        (t0, t1) = (self.token_ids, other.token_ids)
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(len(t0), len(t1))
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2
        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return matched // page_size * page_size if page_size > 1 else matched
        matched_tokens = min(matched_tokens, len(self), len(other))
        if page_size == 1:
            return matched_tokens
        return matched_tokens // page_size * page_size

    def child_key(self, page_size: int=1):
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple(((t[j], t[j + 1]) for j in range(page_size)))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str]=None) -> str:
        hash_value = get_hash_str(self[start:end], prior_hash)
        assert isinstance(hash_value, str)
        return hash_value

class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int]=None, priority: int=0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_value: Optional[torch.Tensor] = None
        self.write_through_pending_id: Optional[int] = None
        self.hash_value: Optional[List[str]] = None
        self.priority = priority
        self.num_prefix_tokens = 0
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        self.host_ref_counter += 1

    def release_host(self):
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError('Host reference counter is already zero.')

    def get_last_hash_value(self) -> Optional[str]:
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time

class RadixCache(SessionRadixCacheMixin, KVCacheEventMixin, BasePrefixCache):

    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.enable_session_radix_cache = params.enable_session_radix_cache
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()
        self.kv_event_queue = []
        if params.enable_metrics:
            self.init_metrics_collector()
        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)
        self._pace_tracker = None
        if self.eviction_policy == 'pace':
            cfg = getattr(params, 'pace_config', None) or {}
            self._pace_tracker = SessionPaceTracker(alpha=cfg.get('alpha', 0.3), ema_floor=cfg.get('ema_floor', 1.0), beta=cfg.get('beta', 1.0), tau_abs=cfg.get('tau_abs', 5.0), k=cfg.get('k', 2.0))
            self._pace_tracker.cache_tokens = float(getattr(params, 'max_total_tokens', 0) or 0)
            os.environ.setdefault('PACE_CACHE_TOKENS', str(getattr(params, 'max_total_tokens', 0) or 0))
            self.eviction_strategy = PaceAwareStrategy(tracker=self._pace_tracker, mode=cfg.get('mode', 'per_token'), p0=cfg.get('p0', 24000.0), lam=cfg.get('lam', 1.0))
        if os.environ.get('PACE_QUOTA', '0') == '1' and self.eviction_policy == 'pace':
            from sglang.srt.mem_cache.evict_policy import QuotaStrategy
            self._quota_strategy = QuotaStrategy(self._pace_tracker, self._session_leaves if hasattr(self, '_session_leaves') else {}, 1.0)
            self.eviction_strategy = self._quota_strategy
        if self.eviction_policy not in ('pace', 'ece', 'arc', 'lecar'):
            self._pace_tracker = None
        self.evictable_leaves = set()
        self._ruei = None
        self.reset()
        if self.eviction_policy == 'pace' and os.environ.get('PACE_RUEI', '0') == '1':
            self._ruei = RhythmUtilityEvictionIndex(self, self.eviction_strategy, time.perf_counter())

    @classmethod
    def create_simulated(self, disable: bool=False, mock_allocator: Optional[Any]=None, page_size: int=1, enable_kv_cache_events: bool=False) -> RadixCache:
        params = CacheInitParams(disable=disable, req_to_token_pool=None, token_to_kv_pool_allocator=mock_allocator, page_size=page_size, enable_kv_cache_events=enable_kv_cache_events)
        return RadixCache(params)

    def reset(self):
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=array('q'), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        ruei = getattr(self, '_ruei', None)
        if ruei is not None:
            ruei.reset(time.perf_counter())
        self._reset_session_radix_state()
        self._empty_match_result = MatchResult(device_indices=torch.empty((0,), dtype=torch.int64, device=self.device), last_device_node=self.root_node, last_host_node=self.root_node, best_match_node=self.root_node)
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        (key, _) = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result
        (value, last_node) = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices
        return MatchResult(device_indices=value, last_device_node=last_node, last_host_node=last_node, best_match_node=last_node)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)
        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked
        (key, value) = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[:len(key)]
        else:
            value = torch.tensor(key.token_ids[:len(key)], dtype=torch.int64)
        (prefix_len, last_node) = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len, last_device_node=last_node)

    def cache_finished_req(self, req: Req, is_insert: bool=True, *, kv_len_to_handle: int):
        if self.disable_finished_insert:
            is_insert = False
        if self._pace_tracker is not None:
            sid = getattr(req, 'session_id', None)
            if sid is not None:
                time_stats = getattr(req, 'time_stats', None)
                arrival = getattr(time_stats, 'scheduler_recv_time', None)
                prompt_len = len(req.origin_input_ids) if hasattr(req, 'origin_input_ids') else 0
                self._pace_tracker.on_finish(sid, arrival, time.perf_counter(), prompt_len=prompt_len)
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :kv_len_to_handle]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :len(token_ids)]
        radix_key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)
        if is_insert:
            priority = getattr(req, 'priority', 0) or 0
            result = self.insert(InsertParams(key=radix_key, value=values, priority=priority))
            session_leaf = result.last_device_node
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len:result.prefix_len])
        else:
            session_leaf = None
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len:key_len])
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])
        self._tag_session_leaf(req, radix_key, node=session_leaf)
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        if self.disable:
            return
        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :len(token_ids)]
        radix_key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle).page_aligned(self.page_size)
        values = kv_indices[:len(radix_key)].to(dtype=torch.int64, copy=True)
        result = self.insert(InsertParams(key=radix_key, value=values, chunked=chunked, priority=getattr(req, 'priority', 0) or 0))
        new_prefix_len = result.prefix_len
        self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len:new_prefix_len])
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        (new_indices, new_last_node) = (match_result.device_indices, match_result.last_device_node)
        assert len(new_indices) == len(radix_key), f'len(new_indices)={len(new_indices)!r}, len(radix_key)={len(radix_key)!r}'
        self.req_to_token_pool.write((req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))), new_indices[req.cache_protected_len:])
        req.cache_protected_len = len(new_indices)
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat([new_indices, kv_indices[len(new_indices):]])
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node
        self._tag_session_leaf(req, radix_key, node=new_last_node)

    def pretty_print(self):
        self._print_helper(self.root_node, 0)

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        _use_ruei = self._ruei is not None
        _candidate_count = len(self.evictable_leaves)
        leaves = () if _use_ruei else list(self.evictable_leaves)
        if hasattr(self.eviction_strategy, 'begin_round'):
            _begin_set = (() if hasattr(self.eviction_strategy, '_incremental_index') else list(self.evictable_leaves)) if _use_ruei and (not leaves) else leaves
            self.eviction_strategy.begin_round(_begin_set)
        if _use_ruei:
            self._ruei.prepare(start_time)
            eviction_heap = None
        else:
            eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
            heapq.heapify(eviction_heap)
        if self._pace_tracker is not None:
            pass
        num_evicted = 0
        while num_evicted < num_tokens:
            if _use_ruei:
                x = self._ruei.pop(time.perf_counter())
                if x is None:
                    break
            else:
                if not eviction_heap:
                    break
                (_priority, x) = heapq.heappop(eviction_heap)
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            parent = x.parent
            if self._pace_tracker is not None:
                self._aggregate_owners_to_parent(x, parent)
            self._delete_leaf(x)
            if not _use_ruei and len(parent.children) == 0 and (parent.lock_ref == 0):
                new_priority = self.eviction_strategy.get_priority(parent)
                heapq.heappush(eviction_heap, (new_priority, parent))
            self._record_remove_event(x)
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _aggregate_owners_to_parent(self, node: TreeNode, parent: TreeNode) -> None:
        owners = getattr(node, 'session_ids', None)
        if not owners or parent is self.root_node:
            return
        self._ensure_session_radix_state()
        parent_owners = getattr(parent, 'session_ids', None)
        if parent_owners is None:
            parent_owners = set()
            parent.session_ids = parent_owners
        for sid in owners:
            parent_owners.add(sid)
            self._session_leaves[sid].add(parent)
        if node.last_access_time > parent.last_access_time:
            parent.last_access_time = node.last_access_time

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(self, node: TreeNode, params: Optional[DecLockRefParams]=None) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)
        original_node = node
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert node is self.root_node, 'This request holds the node from another tree'
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for (_, child) in node.children.items():
                values.append(child.value)
                _dfs_helper(child)
        _dfs_helper(self.root_node)
        return torch.cat(values)

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time
        child_key = key.child_key(self.page_size)
        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]
                if len(key):
                    child_key = key.child_key(self.page_size)
        return (value, node)

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.num_prefix_tokens = child.num_prefix_tokens
        child.num_prefix_tokens = child.num_prefix_tokens + split_len
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        (new_node.hash_value, child.hash_value) = split_node_hash_value(child.hash_value, split_len, self.page_size)
        self._ruei_sync_node(child)
        self._ruei_sync_node(new_node)
        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool=False):
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(self, node: TreeNode, key: RadixKey, value, priority: int=0, chunked: bool=False):
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return (0, node)
        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = key.child_key(self.page_size)
        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.num_prefix_tokens = node.num_prefix_tokens + len(node.key)
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            self._record_store_event(new_node)
            node = new_node
        return (total_prefix_length, node)

    def _print_helper(self, node: TreeNode, indent: int):
        stack = [(node, indent)]
        while stack:
            (current_node, current_indent) = stack.pop()
            for (key, child) in current_node.children.items():
                stack.append((child, current_indent + 2))
                assert key == child.key.child_key(self.page_size), f'key={key!r}, child.key.child_key(self.page_size)={child.key.child_key(self.page_size)!r}'

    def _delete_leaf(self, node):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f'parent does not have child key, {key}'
        self._discard_session_leaf(node)
        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._ruei_sync_node(node)
        self._update_leaf_status(node.parent)

    def _ruei_sync_node(self, node: TreeNode) -> None:
        ruei = getattr(self, '_ruei', None)
        if ruei is not None:
            ruei.sync_node(node)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            self._ruei_sync_node(node)
            return
        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                self._ruei_sync_node(node)
                return
        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)
        self._ruei_sync_node(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size
from sglang.srt.mem_cache.incremental_hooks import install as _install_incremental
from sglang.srt.mem_cache.evict_policy import QuotaStrategy
_install_incremental(SessionPaceTracker, QuotaStrategy, RhythmUtilityEvictionIndex)
from sglang.srt.mem_cache.tp_clock import install_cache as _install_tp_clock
from sglang.srt.mem_cache import evict_policy as _tp_policy, ruei_index as _tp_index
_install_tp_clock(sys.modules[__name__], _tp_policy, _tp_index)
