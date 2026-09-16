from typing import Callable, Iterable


class StateMachine:
    _names: list[str]
    _callbacks: list[Callable | None]
    _cur_state: int

    def __init__(self, states: Iterable[str], init: int = 0):
        self._names = []
        self._callbacks = []
        for state in states:
            self._names.append(state)
            self._callbacks.append(None)
        if len(self._names) <= init:
            raise IndexError("Index out of range")
        self._cur_state = init

    def set_callback(self, state: str | int, callback: Callable):
        if isinstance(state, str):
            self._callbacks[self._names.index(state)] = callback
        elif isinstance(state, int):
            self._callbacks[state] = callback
        else:
            raise TypeError("State must be a string or an integer")

    @property
    def state(self):
        return self._names[self._cur_state]

    @state.setter
    def state(self, state: str | int):
        if isinstance(state, str):
            self._cur_state = self._names.index(state)
        elif isinstance(state, int):
            self._cur_state = state
        else:
            raise TypeError("State must be a string or an integer")

    def run(self, **args):
        cur_func = self._callbacks[self._cur_state]
        if cur_func is None:
            return None
        return cur_func(**args)

    def next(self):
        raise NotImplementedError

    def prev(self):
        raise NotImplementedError

    def reset(self, state=0):
        self.state = state


class StepperStateMachine(StateMachine):
    def next(self):
        if self._cur_state < len(self._names) - 1:
            self._cur_state += 1

    def prev(self):
        if self._cur_state > 0:
            self._cur_state -= 1


class LoopStateMachine(StateMachine):
    def next(self):
        self._cur_state = (self._cur_state + 1) % len(self._names)

    def prev(self):
        self._cur_state = (self._cur_state - 1) % len(self._names)
