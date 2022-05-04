from typing import Optional, List, Callable, Tuple, Iterable

from fedot.core.log import default_log, Log

ConditionType = Callable[[], bool]
ConditionEntryType = Tuple[ConditionType, Optional[str]]


class AggregateCondition:
    """Represents sequence of ordinary conditions with logging.
    All composed conditions are combined with reduce function on booleans.

    By the default 'any' is used, so in this case CompositeCondition is True
    if any of the composed conditions is True. The message corresponding
    to the actual fired condition is logged (if it was provided)."""

    def __init__(self, log: Log = None, conditions_reduce: Callable[[Iterable[bool]], bool] = any):
        self._reduce = conditions_reduce
        self._conditions: List[ConditionEntryType] = []
        self._log = log or default_log(__name__)

    def add_condition(self, condition: ConditionType, log_msg: Optional[str] = None) -> 'AggregateCondition':
        """Builder-like method for adding conditions."""
        self._conditions.append((condition, log_msg))
        return self

    def __bool__(self):
        return self()

    def __call__(self) -> bool:
        return self._reduce(map(self._check_condition, self._conditions))

    def _check_condition(self, entry: ConditionEntryType) -> bool:
        cond, msg = entry
        res = cond()
        if res and msg:
            self._log.info(msg)
        return res