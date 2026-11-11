# PACE 会话节奏、超期静默、有界试用及占用压力；LRU 仅作为结构性回退。
# 依赖与归档相兼容的 SGLang；不是独立完整推理引擎。
from __future__ import annotations
import json
import math
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple, Union
if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

class EvictionStrategy(ABC):

    @abstractmethod
    def get_priority(self, node: 'TreeNode') -> Union[float, Tuple]:
        pass

class LRUStrategy(EvictionStrategy):

    def get_priority(self, node: 'TreeNode') -> float:
        return node.last_access_time

class SessionPaceTracker:

    def __init__(self, alpha: float=0.3, ema_floor: float=1.0, beta: float=1.0, tau_abs: float=5.0, k: float=2.0):
        self.alpha = alpha
        self.ema_floor = ema_floor
        self.beta = beta
        self.tau_abs = tau_abs
        self.k = k
        self.k_online = os.environ.get('PACE_K_ONLINE', '0') == '1'
        from collections import deque as _dq
        self._n1_overdues = _dq(maxlen=1024)
        self._state: dict = {}
        self.ruei_enabled = os.environ.get('PACE_RUEI', '0') == '1'
        self._ruei_dirty: set = set()
        self._ruei_n0_epoch = 0
        self.turn_decay = float(os.environ.get('PACE_TURN_DECAY', '0'))
        self.turn_threshold = int(os.environ.get('PACE_TURN_THRESHOLD', '3'))
        self.dual_ema = os.environ.get('PACE_DUAL_EMA', '0') == '1'
        self.alpha_fast = float(os.environ.get('PACE_ALPHA_FAST', '0.6'))
        self.gap_var = os.environ.get('PACE_GAP_VAR', '0') == '1'
        self.owner_discount = int(os.environ.get('PACE_OWNER_DISCOUNT', '0') or 0)
        self.hard_quota = os.environ.get('PACE_HARD_QUOTA', '0') == '1'
        self.n0_warmup = int(os.environ.get('PACE_N0_WARMUP', '64') or 64)
        self.lambda_floor = float(os.environ.get('PACE_LAMBDA_FLOOR', '0') or 0)
        self.tau_cv = os.environ.get('PACE_TAU_CV', '0') == '1'
        self.imminence = os.environ.get('PACE_IMMINENCE', '0') == '1'
        self.cold_start = os.environ.get('PACE_COLD_START', '0') == '1'
        _seed = os.environ.get('PACE_COLD_SEED')
        self.cold_seed = float(_seed) if _seed is not None else None
        self.n0_mode = os.environ.get('PACE_N0_MODE', '')
        self.n0_online = os.environ.get('PACE_N0_ONLINE', '0').strip()
        if self.n0_online == '0':
            self.n0_online = ''
        from collections import deque
        self.shrink = os.environ.get('PACE_SHRINK', '0') == '1'
        self.shrink_kappa = float(os.environ.get('PACE_SHRINK_KAPPA', '1.0'))
        self._pop_gaps = deque(maxlen=2048)
        from collections import deque
        self._n0_ret_ages = deque(maxlen=1024)
        self.ttl_memo = os.environ.get('PACE_TTL_MEMO', '0') == '1'
        self._n0_ttl_version = 0
        self._n0_ttl_cache = None
        self._n0_ret_lens = deque(maxlen=1024)
        self._n0_risk = 0
        self._tok_times = deque(maxlen=512)
        self._tok_cum = 0
        self.cache_tokens = float(os.environ.get('PACE_CACHE_TOKENS', '0') or 0)
        self.n0_ttl = float(os.environ.get('PACE_N0_TTL', '60.0'))
        self.n0_cap = float(os.environ.get('PACE_N0_CAP', '0.02'))
        self.n0_cap = float(os.environ.get('PACE_N0_CAP', '0.02'))
        self._n0_edges = None
        self._n0_rates = None
        _hz_path = os.environ.get('PACE_N0_HAZARD')
        if _hz_path and os.path.exists(_hz_path):
            with open(_hz_path) as _hz_f:
                _hz = json.load(_hz_f)
            self._n0_edges = _hz['edges']
            self._n0_rates = _hz['rates']
        if self.n0_mode:
            pass
        self.turns_boost = os.environ.get('PACE_TURNS_BOOST', '0') == '1'
        self._tb_scale = float(os.environ.get('PACE_TURNS_SCALE', '1.0'))
        self._tb_table = (1.0, 1.88, 2.05, 2.15, 2.29, 2.39, 2.4, 2.47, 2.51)
        self._tb_cap = 2.51
        self.global_ema = float(os.environ['PACE_GLOBAL_EMA']) if os.environ.get('PACE_GLOBAL_EMA') else None
        self.queue_aware = os.environ.get('PACE_QUEUE_AWARE', '0') == '1'
        self.queued_rate = float(os.environ.get('PACE_QUEUED_RATE', '1e6'))
        self.queue_urgency = os.environ.get('PACE_QUEUE_URGENCY', '0') == '1'
        self._queued = set()
        self.qa_debug = os.environ.get('PACE_QUEUE_DEBUG', '0') == '1'

    def set_queued(self, sids) -> None:
        new = set(sids.keys()) if isinstance(sids, dict) else set(sids) if sids else set()
        if self.ruei_enabled:
            old = getattr(self, '_queued', set())
            if new != old:
                for sid in new ^ old:
                    if sid in self._state:
                        self._ruei_dirty.add(sid)
        self._queued = new
        self._qa_setq += 1

    def on_finish(self, sid, arrival_time: float, now: float, prompt_len=0) -> None:
        if self.n0_online == '2':
            self._tok_cum += max(prompt_len, 0)
            self._tok_times.append((now, self._tok_cum))
        st = self._state.get(sid)
        if st is None:
            if self.n0_online == '2':
                self._n0_risk += 1
            init = {'ema': 0.0, 't_complete': now, 'n': 0, 'turns': 1}
            if self.dual_ema:
                init['ema_fast'] = 0.0
            if self.gap_var or self.tau_cv:
                init['mean_gap'] = 0.0
                init['M2'] = 0.0
            self._state[sid] = init
            if self.ruei_enabled:
                self._ruei_dirty.add(sid)
            return
        if arrival_time is not None:
            gap = arrival_time - st['t_complete']
            if gap > 0.0:
                if st['n'] == 0:
                    if self.n0_online and gap < 30 * 86400:
                        self._n0_ret_ages.append(gap)
                        self._n0_ttl_version += 1
                        if self.ruei_enabled:
                            self._ruei_n0_epoch += 1
                        if self.n0_online == '2':
                            self._n0_ret_lens.append(max(prompt_len, 1))
                    st['ema'] = gap
                    if self.dual_ema:
                        st['ema_fast'] = gap
                    if self.gap_var or self.tau_cv:
                        st['mean_gap'] = gap
                        st['M2'] = 0.0
                else:
                    st['ema'] = self.alpha * gap + (1.0 - self.alpha) * st['ema']
                    if self.dual_ema:
                        st['ema_fast'] = self.alpha_fast * gap + (1.0 - self.alpha_fast) * st.get('ema_fast', st['ema'])
                    if self.gap_var or self.tau_cv:
                        old_mean = st['mean_gap']
                        st['mean_gap'] = old_mean + (gap - old_mean) / st['n']
                        st['M2'] = st.get('M2', 0.0) + (gap - old_mean) * (gap - st['mean_gap'])
                st['n'] += 1
                if self.k_online and st['n'] >= 2 and (0 < gap < 30 * 86400):
                    ema_o = max(st.get('ema', 1.0), self.tau_abs)
                    od = max(0.0, (gap - self.beta * st.get('ema', 1.0)) / ema_o)
                    self._n1_overdues.append(min(od, 50.0))
                if self.shrink and 0 < gap < 30 * 86400:
                    self._pop_gaps.append(gap)
        st['t_complete'] = now
        st['turns'] = st.get('turns', 0) + 1
        if self.ruei_enabled:
            self._ruei_dirty.add(sid)

    def ruei_rate_and_deadline(self, sid, now: float, ratio: float) -> Tuple[Optional[float], Optional[float]]:
        st = self._state.get(sid)
        if st is None:
            return (None, None)
        if self.queue_aware and sid in self._queued:
            top = self.queued_rate
            exponent = math.ceil(math.log(top) / math.log(ratio) - 1e-12)
            return (ratio ** exponent, None)

        def quantize_upper(rate: float) -> float:
            if rate <= 0.0:
                return 0.0
            exponent = math.ceil(math.log(rate) / math.log(ratio) - 1e-12)
            return ratio ** exponent
        age = max(0.0, now - st['t_complete'])
        if st['n'] == 0:
            if self.cold_start:
                ema = self.cold_seed if self.cold_seed is not None else self.tau_abs
                return (quantize_upper(1.0 / max(ema, self.ema_floor)), None)
            if self.n0_mode == 'cap':
                if age > self._eff_ttl():
                    return (0.0, None)
                raw = 1.0 / max(age, self.ema_floor)
                clamped = min(raw, self.n0_cap)
                bucket = quantize_upper(clamped)
                next_rate = bucket / ratio
                next_age = max(self.ema_floor, 1.0 / max(next_rate, 1e-300))
                natural = st['t_complete'] + next_age
                horizon = st['t_complete'] + self._eff_ttl()
                return (bucket, min(natural, horizon))
            horizon = None
            if self.n0_mode == 'ttl_only':
                ttl = self._eff_ttl()
                if age > ttl:
                    return (0.0, None)
                horizon = st['t_complete'] + ttl
            rate = 1.0 / max(age, self.ema_floor)
            bucket = quantize_upper(rate)
            next_rate = bucket / ratio
            next_age = max(self.ema_floor, 1.0 / max(next_rate, 1e-300))
            natural = st['t_complete'] + next_age
            return (bucket, min(natural, horizon) if horizon is not None else natural)
        ema = self._shrunk_ema(st) if self.shrink else st['ema']
        if self.global_ema is not None:
            ema = self.global_ema
        ema_eff = max(ema, self.ema_floor)
        tau = max(ema, self.tau_abs)
        if self.tau_cv and st.get('n', 0) > 2 and (st.get('mean_gap', 0) > 0):
            var = st.get('M2', 0.0) / max(st['n'] - 1, 1)
            cv = math.sqrt(max(0.0, var)) / max(st['mean_gap'], self.ema_floor)
            tau *= 1.0 + min(cv, 3.0)
        effective_beta = self.beta
        if self.gap_var and st.get('n', 0) > 1 and (st.get('mean_gap', 0) > 0):
            var = st.get('M2', 0.0) / (st['n'] - 1)
            cv = math.sqrt(max(0.0, var)) / max(st['mean_gap'], self.ema_floor)
            effective_beta *= 1.0 + cv
        static_factor = 1.0
        if self.dual_ema and st.get('n', 0) > 1:
            trend = st.get('ema_fast', ema) / max(ema, self.ema_floor)
            static_factor *= 1.0 / max(1.0, trend)
        if self.turn_decay > 0:
            excess = max(0, st.get('turns', 0) - self.turn_threshold)
            static_factor *= math.exp(-excess / self.turn_decay)
        overdue = max(0.0, (age - effective_beta * ema) / tau)
        _lam = math.exp(-overdue ** self._eff_k())
        if self.lambda_floor > 0:
            _lam = max(_lam, self.lambda_floor)
        rate = static_factor * _lam / ema_eff
        bucket = quantize_upper(rate)
        next_rate = bucket / ratio
        target_survival = next_rate * ema_eff / max(static_factor, 1e-300)
        if target_survival <= 0.0:
            return (bucket, None)
        target_survival = min(target_survival, 1.0 - 1e-15)
        target_overdue = (-math.log(target_survival)) ** (1.0 / self._eff_k())
        deadline = st['t_complete'] + effective_beta * ema + tau * target_overdue
        return (bucket, deadline)

    def _n0_hazard(self, age: float):
        edges = self._n0_edges
        rates = self._n0_rates
        if not edges or not rates:
            return None
        if age >= edges[-1]:
            return None
        for i in range(len(edges) - 1):
            if edges[i] <= age < edges[i + 1]:
                w = edges[i + 1] - edges[i]
                r0 = rates[i]
                r1 = rates[i + 1] if i + 1 < len(rates) else rates[i]
                frac = (age - edges[i]) / w if w > 0 else 0.0
                return r0 + (r1 - r0) * frac
        return rates[-1]

    def _ols_ttl(self):
        ages = sorted(self._n0_ret_ages)
        if len(ages) < 64 or self._n0_risk < 64:
            return None
        import math
        R = sum(self._n0_ret_lens) / len(self._n0_ret_lens) if self._n0_ret_lens else 1.0
        P = 0.0
        if len(self._tok_times) >= 2 and self.cache_tokens > 0:
            (t0, c0) = self._tok_times[0]
            (t1, c1) = self._tok_times[-1]
            if t1 - t0 > 1.0:
                rate = max(0.0, (c1 - c0) / (t1 - t0))
                P = rate / self.cache_tokens
        if P <= 0:
            P = 5500.0 / max(self.cache_tokens, 1.0)
        edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        n = len(ages)
        (best_t, best_j) = (None, -1e+18)
        for t in edges:
            if t > ages[-1] * 2:
                break
            k = 0
            (lo, hi) = (0, n - 1)
            while lo <= hi:
                mid = (lo + hi) // 2
                if ages[mid] <= t:
                    k = mid + 1
                    lo = mid + 1
                else:
                    hi = mid - 1
            F = k / n
            J = F * R - (1.0 - F) * t * P
            if J > best_j:
                (best_j, best_t) = (J, t)
        return float(best_t) if best_t is not None else None

    def _eff_k(self) -> float:
        if not self.k_online or len(self._n1_overdues) < 64:
            return self.k
        xs = sorted(self._n1_overdues)
        p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
        if p90 <= 1.05:
            return self.k
        import math as _m
        k_hat = _m.log(2.3) / _m.log(p90)
        return max(0.3, min(3.0, k_hat))

    def _shrunk_ema(self, st):
        if not self.shrink or len(self._pop_gaps) < 64:
            return st['ema']
        xs = sorted(self._pop_gaps)
        mu_pop = xs[len(xs) // 2]
        n = max(st.get('n', 0), 1)
        return (n * st['ema'] + self.shrink_kappa * mu_pop) / (n + self.shrink_kappa)

    def _eff_ttl(self) -> float:
        if not self.n0_online or len(self._n0_ret_ages) < getattr(self, 'n0_warmup', 64):
            return self.n0_ttl
        memo = getattr(self, 'ttl_memo', False)
        if memo and self._n0_ttl_cache is not None and (self._n0_ttl_cache[0] == self._n0_ttl_version):
            return self._n0_ttl_cache[1]
        xs = sorted(self._n0_ret_ages)
        value = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
        if memo:
            self._n0_ttl_cache = (self._n0_ttl_version, value)
        return value

    def _turns_factor(self, turns: int) -> float:
        if not self.turns_boost:
            return 1.0
        t = turns if turns >= 1 else 1
        base = self._tb_table[t - 1] if t <= len(self._tb_table) else self._tb_cap
        return 1.0 + (base - 1.0) * self._tb_scale

    def reuse_rate(self, sid, now: float) -> Optional[float]:
        if self.queue_aware and sid in self._queued:
            self._qa_boost += 1
            return self.queued_rate
        st = self._state.get(sid)
        if st is None:
            return None
        age = now - st['t_complete']
        if st['n'] == 0:
            if self.n0_mode == 'ttl_only':
                if age > self._eff_ttl():
                    return None
                return 1.0 / max(age, self.ema_floor)
            if self.n0_mode == 'cap':
                if age > self._eff_ttl():
                    return None
                return min(1.0 / max(age, self.ema_floor), self.n0_cap)
            if self.n0_mode == 'cap':
                if age > self.n0_ttl:
                    return None
                return min(1.0 / max(age, self.ema_floor), self.n0_cap)
            if self.n0_mode == 'hazard':
                h = self._n0_hazard(max(0.0, age))
                if h is None:
                    return None
                return h
            if self.cold_start:
                ema = self.cold_seed if self.cold_seed is not None else self.tau_abs
                ema_eff = max(ema, self.ema_floor)
                if self.imminence:
                    return 1.0 / max(ema - age, self.ema_floor)
                return 1.0 / ema_eff
            r = 1.0 / max(age, self.ema_floor)
            return r
        ema = self._shrunk_ema(st)
        if self.global_ema is not None:
            ema = self.global_ema
        ema_eff = max(ema, self.ema_floor)
        tau = max(ema, self.tau_abs)
        effective_beta = self.beta
        if self.gap_var and st.get('n', 0) > 1 and (st.get('mean_gap', 0) > 0):
            var = st.get('M2', 0.0) / (st['n'] - 1)
            cv = math.sqrt(max(0.0, var)) / max(st['mean_gap'], self.ema_floor)
            effective_beta = self.beta * (1.0 + cv)
        overdue = max(0.0, (age - effective_beta * ema) / tau)
        p_alive = math.exp(-overdue ** self.k)
        if self.dual_ema and st.get('n', 0) > 1:
            ema_fast = st.get('ema_fast', ema)
            ema_slow = ema
            if ema_slow > 0:
                trend = ema_fast / ema_slow
                p_trend = 1.0 / max(1.0, trend)
                p_alive = p_alive * p_trend
        if self.turn_decay > 0:
            turns = st.get('turns', 0)
            excess = max(0, turns - self.turn_threshold)
            p_turn = math.exp(-excess / self.turn_decay)
            p_alive = p_alive * p_turn
        r = p_alive / ema_eff
        r *= self._turns_factor(st.get('turns', 1))
        r *= self._turns_factor(st.get('turns', 1))
        if self.imminence:
            r = p_alive / max(ema - age, self.ema_floor)
        return r

    def get_p_alive(self, sid, now: float) -> float:
        st = self._state.get(sid)
        if st is None:
            return 0.0
        age = now - st['t_complete']
        if st['n'] == 0:
            return 1.0
        ema = st['ema']
        tau = max(ema, self.tau_abs)
        effective_beta = self.beta
        if self.gap_var and st.get('n', 0) > 1 and (st.get('mean_gap', 0) > 0):
            var = st.get('M2', 0.0) / (st['n'] - 1)
            cv = math.sqrt(max(0.0, var)) / max(st['mean_gap'], self.ema_floor)
            effective_beta = self.beta * (1.0 + cv)
        overdue = max(0.0, (age - effective_beta * ema) / tau)
        p_alive = math.exp(-overdue ** self.k)
        if self.dual_ema and st.get('n', 0) > 1:
            ema_fast = st.get('ema_fast', ema)
            ema_slow = ema
            if ema_slow > 0:
                trend = ema_fast / ema_slow
                p_alive = p_alive * (1.0 / max(1.0, trend))
        if self.turn_decay > 0:
            turns = st.get('turns', 0)
            excess = max(0, turns - self.turn_threshold)
            p_alive = p_alive * math.exp(-excess / self.turn_decay)
        return p_alive

    def pop(self, sid) -> None:
        self._state.pop(sid, None)
        if self.ruei_enabled:
            self._ruei_dirty.add(sid)

class PaceAwareStrategy(EvictionStrategy):

    def __init__(self, tracker: Optional[SessionPaceTracker]=None, mode: str='per_token', p0: float=24000.0, lam: float=1.0):
        self.tracker = tracker if tracker is not None else SessionPaceTracker()
        self.mode = (mode or 'per_token').lower()
        self.p0 = p0
        self.lam = lam
        self.ema_floor = self.tracker.ema_floor
        self.value_floor = float(os.environ.get('PACE_VALUE_FLOOR', '0'))
        self.owner_max = os.environ.get('PACE_OWNER_MAX', '0') == '1'
        self.no_value = os.environ.get('PACE_NO_VALUE', '0') == '1'
        self.dead_reclaim = os.environ.get('PACE_DEAD_RECLAIM', '0') == '1'

    def _value(self, node: 'TreeNode') -> float:
        L = len(node.key)
        if L <= 0:
            return 0.0
        D = getattr(node, 'num_prefix_tokens', 0)
        value_per_token = self.p0 + D + (L - 1) / 2.0
        if self.mode == 'per_node':
            recompute = L * value_per_token
            return 1.0 + self.lam * math.log(1.0 + recompute)
        return value_per_token

    def _reuse_rate(self, node: 'TreeNode', now: float) -> float:
        owners = getattr(node, 'session_ids', None)
        rate = 0.0
        if owners:
            for sid in owners:
                r = self.tracker.reuse_rate(sid, now)
                if r is not None:
                    rate = max(rate, r) if self.owner_max else rate + r
        if rate > 0.0:
            if self.value_floor > 0:
                val = self._value(node)
                min_rate = self.value_floor / max(val, 1.0)
                rate = max(rate, min_rate)
            return rate
        age = time.monotonic() - node.last_access_time
        fallback = 1.0 / max(age, self.ema_floor)
        return fallback

    def get_priority(self, node: 'TreeNode'):
        now = time.perf_counter()
        if getattr(self.tracker, 'queue_urgency', False):
            val = 1.0 if self.no_value else self._value(node)
            pos = self._queued_front_pos(node)
            if pos is not None:
                return (1, val * (1.0 / (pos + 1.0)))
            return (0, val * self._reuse_rate(node, now))
        if self.dead_reclaim:
            owners = getattr(node, 'session_ids', None)
            if owners:
                for sid in owners:
                    r = self.tracker.reuse_rate(sid, now)
                    if r is None or r <= 0.0:
                        return (-1, getattr(node, 'last_access_time', 0))
            return (0, self._value(node) * self._reuse_rate(node, now))
        if self.no_value:
            return self._reuse_rate(node, now)
        return self._value(node) * self._reuse_rate(node, now)

    def _queued_front_pos(self, node: 'TreeNode'):
        owners = getattr(node, 'session_ids', None)
        q = getattr(self.tracker, '_queued', None)
        if not owners or not q or (not hasattr(q, 'get')):
            return None
        best = None
        for sid in owners:
            pos = q.get(sid)
            if pos is not None and (best is None or pos < best):
                best = pos
        return best

    def owner_p_alive(self, node: 'TreeNode', now: float) -> float:
        owners = getattr(node, 'session_ids', None)
        if not owners:
            return 0.0
        best = 0.0
        for sid in owners:
            p = self.tracker.get_p_alive(sid, now)
            if p > best:
                best = p
        return best

class QuotaStrategy(EvictionStrategy):

    def __init__(self, tracker, session_leaves, total_capacity: float):
        self.is_quota = True
        self.tracker = tracker
        self.session_leaves = session_leaves
        self.B = float(total_capacity)
        self.ignore_occupancy = os.environ.get('PACE_QUOTA_IGNORE_OCC', '0') == '1'
        self.head_preserve = os.environ.get('PACE_HEAD_PRESERVE', '0') == '1'
        self.budget_floor_frac = float(os.environ.get('PACE_BUDGET_FLOOR', '0') or 0)
        self.ema_floor = getattr(tracker, 'ema_floor', 1.0)
        self._occ = {}
        self._W = 0.0

    def begin_round(self, leaves) -> None:
        self._occ = {}
        now = time.perf_counter()
        for node in leaves:
            owners = getattr(node, 'session_ids', None)
            n_own = len(owners) if owners else 0
            if n_own == 0:
                continue
            share = len(node.value) / float(n_own)
            for sid in owners:
                self._occ[sid] = self._occ.get(sid, 0.0) + share
        W = 0.0
        for sid in self._occ:
            r = self.tracker.reuse_rate(sid, now)
            if r is not None and r > 0.0:
                W += r
        self._W = W

    def _ratio(self, sid, now):
        occ = self._occ.get(sid)
        if occ is None:
            return 0.0
        r = self.tracker.reuse_rate(sid, now)
        if r is None or r <= 0.0:
            return float('inf')
        if self._W <= 0.0 or self.B <= 0.0:
            return 0.0
        budget = self.B * r / self._W
        if self.budget_floor_frac > 0:
            budget = max(budget, self.budget_floor_frac * self.B)
        if budget == 0.0:
            return float('inf') if (1.0 if self.ignore_occupancy else occ) > 0.0 else 0.0
        return (1.0 if self.ignore_occupancy else occ) / budget

    def get_priority(self, node: 'TreeNode'):
        owners = getattr(node, 'session_ids', None)
        if not owners:
            age = time.monotonic() - node.last_access_time
            return (1.0, 1.0 / max(age, self.ema_floor))
        now = time.perf_counter()
        self._dead_filter = getattr(self, '_dead_filter', None) or os.environ.get('PACE_DEAD_FILTER', '0') == '1'
        worst = None
        for sid in owners:
            rt = self._ratio(sid, now)
            if self._dead_filter and rt == float('inf'):
                continue
            if worst is None or rt > worst:
                worst = rt
        if worst is None:
            if self._dead_filter:
                if getattr(self, 'head_preserve', False):
                    return (float('-inf'), -len(node.key))
                return (float('-inf'), node.last_access_time)
            worst = float('inf')
        od = getattr(self.tracker, 'owner_discount', 0)
        if od > 0:
            denom = float(max(len(owners), 1)) ** float(od)
            return (-worst / denom, node.last_access_time)
        if getattr(self.tracker, 'hard_quota', False) and worst < 1.0:
            return (1.0, node.last_access_time)
        return (-worst, node.last_access_time)

def get_eviction_strategy(policy):
    """仅提供 PACE 及未标记缓存的 LRU 回退，不提供实验基线。"""
    policy = policy.lower()
    if policy == 'pace':
        return PaceAwareStrategy()
    if policy == 'lru':
        return LRUStrategy()
    raise ValueError('此机制包仅支持 pace 和 lru 回退路径')
