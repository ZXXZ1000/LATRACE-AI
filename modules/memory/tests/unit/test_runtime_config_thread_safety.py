from __future__ import annotations

import threading
import time
from typing import Dict, Any

from modules.memory.application import runtime_config as rc


def test_rerank_weights_concurrent_set_and_get():
    allowed = {"alpha_vector", "beta_bm25", "gamma_graph", "delta_recency", "user_boost", "domain_boost", "session_boost"}

    def _writer(idx: int):
        # keep weights summing to 1.0 to satisfy validation
        base = 0.2 + 0.005 * idx
        w = {
            "alpha_vector": base,
            "beta_bm25": base,
            "gamma_graph": base,
            "delta_recency": 1.0 - 3 * base,
        }
        rc.set_rerank_weights(w)

    def _reader(stop_at: float, sink: list):
        while time.time() < stop_at:
            d = rc.get_rerank_weights_override()
            # keys must be subset of allowed and values are floats
            assert set(d.keys()).issubset(allowed)
            for v in d.values():
                assert isinstance(v, float)
            sink.append(d)

    # run writers/readers concurrently
    stop_time = time.time() + 0.2
    reads: list[Dict[str, Any]] = []
    threads = [threading.Thread(target=_reader, args=(stop_time, reads)) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(20):
        _writer(i)
    for t in threads:
        t.join()
    # At least some reads happened
    assert reads, "no reads captured"


def test_graph_params_concurrent_updates_and_reads():
    def _writer(idx: int):
        rc.set_graph_params(
            rel_whitelist=["APPEARS_IN", "DESCRIBES"] if idx % 2 == 0 else ["EQUIVALENCE"],
            max_hops=(idx % 3) + 1,
            neighbor_cap_per_seed=5 + (idx % 5),
            restrict_to_user=(idx % 2 == 0),
            restrict_to_domain=True,
        )

    def _reader(stop_at: float, sink: list):
        while time.time() < stop_at:
            d = rc.get_graph_params_override()
            # Types must be stable
            if "max_hops" in d:
                assert isinstance(d["max_hops"], int)
            if "neighbor_cap_per_seed" in d:
                assert isinstance(d["neighbor_cap_per_seed"], int)
            for k in ("restrict_to_user", "restrict_to_domain"):
                if k in d:
                    assert isinstance(d[k], bool)
            sink.append(d)

    stop_time = time.time() + 0.2
    reads: list[Dict[str, Any]] = []
    threads = [threading.Thread(target=_reader, args=(stop_time, reads)) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(50):
        _writer(i)
    for t in threads:
        t.join()
    assert reads, "no reads captured"
