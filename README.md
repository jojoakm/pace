# PACE: Session-Aware Prefix Cache Management for Conversational LLM Serving

Reference implementation of the PACE eviction mechanism described in the paper
*PACE: Session-Aware Prefix Cache Management for Conversational LLM Serving*.

This repository contains **only** the PACE mechanism. It does not include the
replay harness, benchmark drivers, datasets, experiment logs, or measured
results. It is a patch set that runs on top of a compatible SGLang version
rather than a standalone serving engine.

## Contents

```
PACE_mechanism/
  paper_config.py                      # locked parameters and required PACE switches
  python/sglang/srt/mem_cache/
    evict_policy.py                    # session pace tracking, reuse weight, trial, quota pressure
    ruei_index.py                      # RUEI: geometric bands, crossing deadlines, versioned heap
    incremental_hooks.py               # event accounting and incremental node maintenance
    radix_cache.py                     # radix-cache integration points
    session_radix_cache.py             # session tagging and holder propagation
    tp_clock.py                        # tensor-parallel clock handling
    cache_init_params.py               # cache configuration fields
```

## What the mechanism does

1. **Session-awareness layer.** For each session, the tracker keeps the latest
   completion time, the current silence, and a pace estimate updated by an
   exponential moving average over observed positive return gaps. A reuse
   weight combines the usual return frequency with a discount for silence that
   exceeds the session's grace period. Sessions without an observed return gap
   receive a bounded trial weight.
2. **Eviction-decision layer.** Each session's reuse weight is turned into a
   reference share of the cache and compared with the token occupancy the
   session actually holds on the evictable frontier. Node priority is the
   maximum pressure over the node's holders.
3. **RUEI.** Weights are rounded onto geometric bands, the next band-crossing
   instant is solved in closed form, and crossings are scheduled on a timing
   wheel. A persistent versioned heap keeps node keys current, so decay does
   not require rescoring every candidate at each eviction.

## Usage

The module expects a compatible SGLang source tree. Copy the files under
`PACE_mechanism/python/` onto the matching paths of that tree, then call
`paper_config.configure()` before starting the server.

```python
from PACE_mechanism import paper_config

config = paper_config.configure("WC")   # or "SC"
# config = {"eviction_policy": "pace", "enable_session_radix_cache": True, ...}
```

`configure()` clears any pre-existing `PACE_*` environment variables, applies
the parameter set locked for the named dataset, and enables the required
runtime switches. Model path, GPUs, and cache capacity remain deployment
choices and are not set here.

Note that `configure()` selects the paper configuration. Ablation variants of
the paper are implemented as other switch combinations of the same mechanism
and are intentionally not exposed here.

## Scope and limitations

- This is not a complete reproduction package. It cannot reproduce the numbers
  in the paper without the replay harness and the original workloads.
- The code targets a compatible SGLang version; it is not maintained against
  upstream SGLang releases.
- Only the mechanism is provided. Experiment orchestration, auditing, and
  analysis scripts are out of scope.

## Provenance and licensing

These files are modified excerpts of the SGLang memory-cache layer, which is
distributed under the Apache License 2.0. The upstream license continues to
apply to derived code. No separate license is asserted here for this snapshot;
consult the SGLang project for the license text and attribution requirements.
