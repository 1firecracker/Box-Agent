"""Proactive-owned host bindings and complete-turn lifecycle.

ACP opts into these extensions; the shared Agent service and session contracts
remain independent of Proactive. Transport callbacks stay on borrowed bindings,
never in serializable session state or model-visible diagnostic events.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ...session_context import HostBindings


@dataclass(slots=True)
class ProactiveHostBindings(HostBindings):
    publish_notification: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None


@runtime_checkable
class UserTurnObserverPort(Protocol):
    enabled: bool

    def begin_turn(self, *, user: str, session_id: str, turn_id: str) -> Any: ...
    def end_turn(self, token: Any, *, result: Any, failed: bool) -> None: ...
    def invalidate(self) -> None: ...
    async def drain(self) -> None: ...


@dataclass
class UserTurnResult:
    """The ACP host supplies its final result after all continuation runs."""

    result: Any = None


class ProactiveLifecycle:
    @staticmethod
    def turn_observers(session: Any) -> tuple:
        managed = getattr(session, "plugin_session", None)
        if managed is None:
            return ()
        return tuple(
            observer for observer in managed.activation.registry.get_all(UserTurnObserverPort)
            if observer.enabled
        )

    @staticmethod
    @asynccontextmanager
    async def user_turn(session: Any, *, user: str, session_id: str, turn_id: str):
        observers = ProactiveLifecycle.turn_observers(session)
        tokens = [
            (observer, observer.begin_turn(user=user, session_id=session_id, turn_id=turn_id))
            for observer in observers
        ]
        completed = UserTurnResult()
        failed = True
        try:
            yield completed
            failed = False
        finally:
            for observer, token in reversed(tokens):
                observer.end_turn(token, result=completed.result, failed=failed)

    @staticmethod
    def user_input_started(session: Any) -> None:
        for observer in ProactiveLifecycle.turn_observers(session):
            observer.invalidate()

    @staticmethod
    async def drain_turn_observers(session: Any) -> None:
        for observer in ProactiveLifecycle.turn_observers(session):
            await observer.drain()
