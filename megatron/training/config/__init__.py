# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Compatibility shim for Megatron-Bridge which imports from megatron.training.config.
# The sj/cudagraph-on-ultra branch has these config classes at the megatron.training.*
# level (flat layout), but Megatron-Bridge expects the newer megatron.training.config.*
# subpackage layout.

import os
from dataclasses import dataclass, field
from typing import Literal

from megatron.training.training_config import (
    CheckpointConfig,
    LoggerConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.training.common_config import (
    ProfilingConfig,
    RNGConfig,
)
from megatron.training.resilience_config import (
    RerunStateMachineConfig,
    StragglerDetectionConfig,
)


@dataclass
class DistributedInitConfig:
    """Configuration settings for distributed training initialization.

    Stub for sj/cudagraph-on-ultra compatibility — this class was added to
    Megatron-LM after the CG branch was cut.
    """

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    distributed_timeout_minutes: int = 10
    align_grad_reduce: bool = True
    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    lazy_mpu_init: bool = False
    use_megatron_fsdp: bool = False
    use_torch_fsdp2: bool = False
    use_gloo_process_groups: bool = False
    use_tp_pp_dp_mapping: bool = False
