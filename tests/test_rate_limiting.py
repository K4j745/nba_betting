# tests/test_rate_limiting.py
import time
import pytest
from unittest.mock import patch
from data_fetcher import _rand_sleep
from config import REQUEST_DELAY_MIN, REQUEST_DELAY_MAX


def test_rand_sleep_within_bounds():
    """Sleep musi być w przedziale 0.6–1.0s — weryfikujemy przez monkey-patch time.sleep."""
    sleep_calls = []

    with patch("data_fetcher.time.sleep", side_effect=lambda x: sleep_calls.append(x)):
        for _ in range(20):
            _rand_sleep()

    assert len(sleep_calls) == 20
    for delay in sleep_calls:
        assert REQUEST_DELAY_MIN <= delay <= REQUEST_DELAY_MAX, \
            f"Delay {delay:.3f}s poza zakresem [{REQUEST_DELAY_MIN}, {REQUEST_DELAY_MAX}]"
