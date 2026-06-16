from benchmark.cvbench import CVBench
from benchmark.mmvet import MMVet
from benchmark.vilp import VILP

BENCHMARK_MAP = {
    "MMVet": MMVet,
    "CVBench": CVBench,
    "VILP": VILP,
}

BENCHMARK_TYPE = {
    "MMVet": "FREE_FORM",
    "CVBench": "MULTI_CHOICE",
    "VILP": "FREE_FORM",
}
