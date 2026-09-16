"""Timing helpers for allstar toilet runtime scripts."""

import time


class RateLimiter:
    """Keep a control loop close to a target frequency."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz

    def sleep(self) -> None:
        cur_time = time.time()
        next_wakeup_time = self.last_time + self.sleep_duration
        if cur_time < next_wakeup_time:
            time.sleep(next_wakeup_time - cur_time)
        self.last_time = self.last_time + self.sleep_duration
        while self.last_time + self.sleep_duration < time.time():
            self.last_time += self.sleep_duration
