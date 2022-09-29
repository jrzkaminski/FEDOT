from __future__ import annotations

from collections import UserList
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from itertools import chain

from typing import TYPE_CHECKING, Any, Dict, List, Iterable, Optional, Union, Callable

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
    from fedot.core.optimisers.opt_history import Individual, OptHistory

individuals_pool: ContextVar[Dict[str, Individual]] = ContextVar('individuals_pool')


@contextmanager
def history_generations_context(history: OptHistory):
    # Create a context variable that persists until exiting the context.
    token = individuals_pool.set({})
    try:
        yield
    finally:
        # Add the accumulated `individuals_pool` to history.
        uids = set()
        for ind in chain(*history.individuals):
            uids.add(ind.uid)
            for operator in ind.operators_from_prev_generation[1:]:
                uids.update(i.uid for i in operator.parent_individuals)
        history.individuals_pool = {uid: ind for uid, ind in individuals_pool.get().items() if uid in uids}
        # Clean the context variable.
        individuals_pool.reset(token)


class IndividualsList(UserList):
    def __init__(self, iterable: Optional[Iterable[Individual]] = None):
        iterable = iterable or ()
        try:
            self.individuals_pool = individuals_pool.get()
        except LookupError:
            self.individuals_pool = {}
        uids = self.__iterable_to_individuals_pool(iterable)
        super().__init__(uids)

    def _append_to_individuals_pool(self, individual: Individual):
        self.individuals_pool[individual.uid] = individual
        for parent_operator in individual.operators_from_prev_generation:
            parent_operator.parent_individuals.individuals_pool = self.individuals_pool

    def __iterable_to_individuals_pool(self, iterable: Iterable[Individual]) -> List[str]:
        uids = []
        for item in iterable:
            self._append_to_individuals_pool(item)
            uids.append(item.uid)
        return uids

    def __setitem__(self, index, item: Union[Individual, Iterable[Individual]]):
        if isinstance(index, slice):
            uids = self.__iterable_to_individuals_pool(item)
            self.data[index] = uids
        else:
            self._append_to_individuals_pool(item)
            self.data[index] = item.uid

    def __getitem__(self, index) -> Union[Individual, Iterable[Individual]]:
        if isinstance(index, slice):
            new = copy(self)
            new.data = new.data[index]
            return new
        else:
            uid = self.data[index]
            return self.individuals_pool[uid]

    def __contains__(self, item: Individual):
        return item.uid in self.data

    def insert(self, index, item: Individual):
        self._append_to_individuals_pool(item)
        self.data.insert(index, item.uid)

    def append(self, item: Individual):
        self._append_to_individuals_pool(item)
        self.data.append(item.uid)

    def extend(self, other: Union[IndividualsList, Iterable[Individual]]):
        if isinstance(other, type(self)):
            self.individuals_pool.update(other.individuals_pool)
            self.data.extend(other.data)
        else:
            uids = self.__iterable_to_individuals_pool(other)
            self.data.extend(uids)

    def remove(self, item: Individual):
        self.data.remove(item.uid)

    def set_data(self, other: Union[IndividualsList, Iterable[Individual]]) -> IndividualsList:
        if isinstance(other, type(self)):
            self.data = other.data
        else:
            uids = self.__iterable_to_individuals_pool(other)
            super().__init__(uids)
        return self

    def copy_with_data(self, other: Union[IndividualsList, Iterable[Individual]]) -> IndividualsList:
        new = copy(self)
        new.set_data(other)
        return new

    def apply(self, func: Callable[[PopulationT], PopulationT], inplace: bool = False) -> IndividualsList:
        result = func(list(self))
        if inplace:
            return self.set_data(result)
        else:
            return self.copy_with_data(result)


class Generation(IndividualsList):
    def __init__(self, iterable: Union[Iterable[Individual], IndividualsList], generation_num: int, label: str = '',
                 metadata: Optional[Dict[str, Any]] = None):
        self.generation_num = generation_num
        self.label = label
        self.metadata: Dict[str, Any] = metadata or {}
        super().__init__(iterable)

    def _append_to_individuals_pool(self, individual: Individual):
        individual.set_native_generation(self.generation_num)
        super()._append_to_individuals_pool(individual)
