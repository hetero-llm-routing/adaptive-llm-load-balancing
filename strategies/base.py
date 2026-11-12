"""The one interface every strategy implements (agreed by the team).

    choose_gpu(request, gpu_states) -> gpu_id

The same function runs unchanged in the real router and in the simulator.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Request:
    request_id: str
    prompt_len: int            # tokens in the prompt (estimated on arrival)
    req_type: str = "chat"     # "chat", "doc" (long document) or "rag"
    arrival_ts: float = 0.0    # seconds since the epoch
    true_output_len: int = None  # known only in experiments (max_tokens with ignore_eos); used by the oracle only


@dataclass
class GpuState:
    queue_len: int = 0         # requests waiting on the node
    running: int = 0           # requests running on the node
    kv_usage: float = 0.0      # 0.0 to 1.0
    extra: Dict = field(default_factory=dict)


def choose_gpu(request: Request, gpu_states: Dict[str, GpuState]) -> str:
    raise NotImplementedError
