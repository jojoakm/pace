"""论文锁定参数与 PACE 必要开关；在启动兼容SGLang之前调用configure。
模型路径、GPU和缓存容量由调用者配置；不含部署地址或运行结果。
"""
import os

PARAMETERS = {'WC': {'PACE_ALPHA': 0.2905284368931027, 'PACE_BETA': 4.707657931813611, 'PACE_TAU_ABS': 73.17983273107147, 'PACE_K': 1.846633796097096, 'PACE_N0_TTL': 78.3781863244774, 'PACE_N0_CAP': 0.009304529385478606}, 'SC': {'PACE_ALPHA': 0.6108184652921604, 'PACE_BETA': 7.660288851760951, 'PACE_TAU_ABS': 6.836885017888096, 'PACE_K': 0.6046680390385137, 'PACE_N0_TTL': 7.623591628738961, 'PACE_N0_CAP': 0.01069850641511325}}

def configure(dataset):
    """启用论文的三层PACE配置，不启用实验变体。"""
    values = PARAMETERS[dataset.upper()]
    for key in tuple(os.environ):
        if key.startswith("PACE_"):
            os.environ.pop(key)
    os.environ.update({key: str(value) for key, value in values.items()})
    os.environ.update(PACE_RUEI="1", PACE_QUOTA="1", PACE_N0_MODE="cap",
                      PACE_N0_ONLINE="1", PACE_TTL_MEMO="1", PACE_RUEI_LAZY_TTL="1",
                      PACE_RUEI_INCREMENTAL="1", PACE_RUEI_RATIO="2", PACE_RUEI_TICK="0.1",
                      PACE_N0_WARMUP="64", PACE_RUEI_WHEEL_SIZE="4096")
    return {"eviction_policy": "pace", "enable_session_radix_cache": True,
            "pace_config": {"alpha": values["PACE_ALPHA"], "beta": values["PACE_BETA"],
                            "tau_abs": values["PACE_TAU_ABS"], "k": values["PACE_K"]}}
