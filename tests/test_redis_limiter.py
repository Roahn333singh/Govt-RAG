"""Shared limiter storage: two app 'instances' on one storage share a budget."""

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter


def test_shared_storage_counts_across_instances():
    storage = MemoryStorage()  # stands in for the shared Redis
    inst_a = FixedWindowRateLimiter(storage)
    inst_b = FixedWindowRateLimiter(storage)
    limit = parse("15/minute")
    results = []
    for i in range(30):
        lim = inst_a if i % 2 == 0 else inst_b
        results.append(lim.hit(limit, "198.51.100.7"))
    assert results.count(True) == 15
    assert results.count(False) == 15


def test_limiter_configured_to_fail_open():
    import main

    assert main.limiter._swallow_errors is True
