from benchmark.cvbench import CVBench
from benchmark.mmvet import MMVet
from benchmark.vilp import VILP
from benchmark.visualcot import VisualCot

BENCHMARK_MAP = {
    "MMVet": MMVet,
    "CVBench": CVBench,
    "VILP": VILP,
    "VisualCot": VisualCot,
}

BENCHMARK_TYPE = {
    "MMVet": "FREE_FORM",
    "CVBench": "MULTI_CHOICE",
    "VILP": "FREE_FORM",
    "VisualCot": "FREE_FORM",
}
