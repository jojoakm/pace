# PACE 张量并行决策时钟接入，不包含实验部署或运行日志。
# 依赖与归档相兼容的 SGLang；不是独立完整推理引擎。
import os
import time
import torch
import torch.distributed as dist

class CycleClock:
    value = None

    def perf_counter(self):
        return time.perf_counter() if self.value is None else self.value

    def monotonic(self):
        return time.monotonic() if self.value is None else self.value

    def __getattr__(self, name):
        return getattr(time, name)
clock = CycleClock()

def install_cache(radix, policy, index):
    if os.environ.get('PACE_TP_CLOCK', '0') != '1':
        return
    radix.time = policy.time = index.time = clock
    radix.TreeNode.__hash__ = lambda node: node.id
    radix.TreeNode.__lt__ = lambda node, other: (node.last_access_time, node.id) < (other.last_access_time, other.id)

def install_scheduler(Scheduler):
    if os.environ.get('PACE_TP_CLOCK', '0') != '1':
        return
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
    original_stamp = SchedulerReqTimeStats.set_scheduler_recv_time

    def stamp(self, ts=None):
        return original_stamp(self, clock.perf_counter() if ts is None else ts)
    SchedulerReqTimeStats.set_scheduler_recv_time = stamp
    original = Scheduler.process_input_requests

    def process(self, requests):
        assert self.ps.pp_size == 1 and (not self.enable_dp_attention)
        if not hasattr(self, '_pace_cycle_tensor'):
            self._pace_cycle_tensor = torch.zeros(1, dtype=torch.float64, device='cpu')
            self._pace_clock_cycles = 0
        if self.tp_group.rank_in_group == 0:
            self._pace_cycle_tensor[0] = time.perf_counter()
        dist.broadcast(self._pace_cycle_tensor, src=self.tp_group.first_rank, group=self.tp_cpu_group)
        clock.value = float(self._pace_cycle_tensor[0])
        self._pace_clock_cycles += 1
        return original(self, requests)
    Scheduler.process_input_requests = process
