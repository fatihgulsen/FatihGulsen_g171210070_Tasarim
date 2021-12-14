import time
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Optional


class TimerError(Exception):
    pass


@dataclass
class Timer:
    timers: ClassVar[Dict[str, list]] = dict()
    name: Optional[str] = None
    text: str = "Gecen Sure: {:0.4f} saniye"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)
    min_text = text + ' *min*'
    max_text = text + ' *max*'
    mean_text = text + ' *mean*'

    def __post_init__(self) -> None:
        if self.name is not None:
            self.timers.setdefault(self.name, [])

    def start(self) -> None:
        if self._start_time is not None:
            raise TimerError(f"Timer calisiyor. stop() kullanarak durdurabilirsiniz")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        if self._start_time is None:
            raise TimerError(f"Timer calismıyor. start() kullarak calistirabilirsiniz")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name].append(elapsed_time)

        return elapsed_time

    def timer_detail(self) -> None:
        self.logger('\n************\n')
        self.logger(self.text.format(sum(self.timers[self.name])))
        self.logger(self.min_text.format(min(self.timers[self.name])))
        self.logger(self.max_text.format(max(self.timers[self.name])))
        self.logger(self.mean_text.format((sum(self.timers[self.name]))/len(self.timers[self.name])))
        self.logger('\n************\n')

