"""Tests for the SessionManager class."""

from unittest.mock import AsyncMock, PropertyMock

from scrapling.core._types import Any
import pytest

from scrapling.spiders.session import SessionManager
from scrapling.spiders.request import Request


class MockSession:  # type: ignore[type-arg]
    """Mock session for testing without actual network calls."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._is_alive = False
        self._started = False
        self._closed = False

    async def __aenter__(self):
        self._is_alive = True
        self._started = True
        return self

    async def __aexit__(self, *args):
        self._is_alive = False
        self._closed = True

    async def fetch(self, url: str, **kwargs):
        pass


class TestSessionManagerInit:
    """Test SessionManager initialization."""

    def test_manager_starts_empty(self):
        """Test that manager starts with no sessions."""
        manager = SessionManager()

        assert len(manager) == 0

    def test_manager_no_default_session_when_empty(self):
        """Test that accessing default_session_id raises when empty."""
        manager = SessionManager()

        with pytest.raises(RuntimeError, match="No sessions registered"):
            _ = manager.default_session_id


class TestSessionManagerAdd:
    """Test SessionManager add functionality."""

    def test_add_single_session(self):
        """Test adding a single session."""
        manager = SessionManager()
        session = MockSession()

        manager.add("test", session)

        assert len(manager) == 1
        assert "test" in manager
        assert manager.session_ids == ["test"]

    def test_first_session_becomes_default(self):
        """Test that first added session becomes default."""
        manager = SessionManager()
        session = MockSession()

        manager.add("first", session)

        assert manager.default_session_id == "first"

    def test_add_multiple_sessions(self):
        """Test adding multiple sessions."""
        manager = SessionManager()

        manager.add("session1", MockSession("s1"))
        manager.add("session2", MockSession("s2"))
        manager.add("session3", MockSession("s3"))

        assert len(manager) == 3
        assert "session1" in manager
        assert "session2" in manager
        assert "session3" in manager

    def test_explicit_default_session(self):
        """Test setting explicit default session."""
        manager = SessionManager()

        manager.add("first", MockSession())
        manager.add("second", MockSession(), default=True)

        assert manager.default_session_id == "second"

    def test_add_duplicate_id_raises(self):
        """Test that adding duplicate session ID raises."""
        manager = SessionManager()
        manager.add("test", MockSession())

        with pytest.raises(ValueError, match="already registered"):
            manager.add("test", MockSession())

    def test_add_returns_self_for_chaining(self):
        """Test that add returns self for method chaining."""
        manager = SessionManager()

        result = manager.add("test", MockSession())

        assert result is manager

    def test_method_chaining(self):
        """Test fluent interface for adding sessions."""
        manager = SessionManager()

        manager.add("s1", MockSession()).add("s2", MockSession()).add("s3", MockSession())

        assert len(manager) == 3

    def test_add_lazy_session(self):
        """Test adding lazy session."""
        manager = SessionManager()

        manager.add("lazy", MockSession(), lazy=True)

        assert "lazy" in manager
        assert "lazy" in manager._lazy_sessions


class TestSessionManagerRemove:
    """Test SessionManager remove/pop functionality."""

    def test_remove_session(self):
        """Test removing a session."""
        manager = SessionManager()
        manager.add("test", MockSession())

        manager.remove("test")

        assert "test" not in manager
        assert len(manager) == 0

    def test_remove_nonexistent_raises(self):
        """Test removing nonexistent session raises."""
        manager = SessionManager()

        with pytest.raises(KeyError, match="not found"):
            manager.remove("nonexistent")

    def test_pop_returns_session(self):
        """Test pop returns the removed session."""
        manager = SessionManager()
        session = MockSession("original")
        manager.add("test", session)

        popped = manager.pop("test")

        assert popped is session
        assert "test" not in manager

    def test_remove_default_updates_default(self):
        """Test that removing default session updates default."""
        manager = SessionManager()
        manager.add("first", MockSession())
        manager.add("second", MockSession())

        assert manager.default_session_id == "first"

        manager.remove("first")

        assert manager.default_session_id == "second"

    def test_remove_lazy_session_cleans_up(self):
        """Test that removing lazy session cleans up lazy set."""
        manager = SessionManager()
        manager.add("lazy", MockSession(), lazy=True)

        manager.remove("lazy")

        assert "lazy" not in manager._lazy_sessions


class TestSessionManagerGet:
    """Test SessionManager get functionality."""

    def test_get_existing_session(self):
        """Test getting an existing session."""
        manager = SessionManager()
        session = MockSession("test")
        manager.add("test", session)

        retrieved = manager.get("test")

        assert retrieved is session

    def test_get_nonexistent_raises_with_available(self):
        """Test getting nonexistent session shows available sessions."""
        manager = SessionManager()
        manager.add("session1", MockSession())
        manager.add("session2", MockSession())

        with pytest.raises(KeyError, match="Available:"):
            manager.get("nonexistent")


class TestSessionManagerContains:
    """Test SessionManager contains functionality."""

    def test_contains_existing(self):
        """Test contains for existing session."""
        manager = SessionManager()
        manager.add("test", MockSession())

        assert "test" in manager

    def test_not_contains_missing(self):
        """Test contains for missing session."""
        manager = SessionManager()
        manager.add("test", MockSession())

        assert "other" not in manager


class TestSessionManagerAsyncContext:
    """Test SessionManager async context manager."""

    @pytest.mark.asyncio
    async def test_start_activates_sessions(self):
        """Test that start activates non-lazy sessions."""
        manager = SessionManager()
        session = MockSession()
        manager.add("test", session)

        await manager.start()

        assert session._is_alive is True
        assert manager._started is True

    @pytest.mark.asyncio
    async def test_start_skips_lazy_sessions(self):
        """Test that start skips lazy sessions."""
        manager = SessionManager()
        eager_session = MockSession("eager")
        lazy_session = MockSession("lazy")

        manager.add("eager", eager_session)
        manager.add("lazy", lazy_session, lazy=True)

        await manager.start()

        assert eager_session._is_alive is True
        assert lazy_session._is_alive is False

    @pytest.mark.asyncio
    async def test_close_deactivates_sessions(self):
        """Test that close deactivates all sessions."""
        manager = SessionManager()
        session = MockSession()
        manager.add("test", session)

        await manager.start()
        assert session._is_alive is True

        await manager.close()
        assert session._is_alive is False
        assert manager._started is False

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test using SessionManager as async context manager."""
        manager = SessionManager()
        session = MockSession()
        manager.add("test", session)

        async with manager:
            assert session._is_alive is True

        assert session._is_alive is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that calling start multiple times is safe."""
        manager = SessionManager()
        session = MockSession()
        manager.add("test", session)

        await manager.start()
        await manager.start()  # Should not raise or double-start

        assert session._started is True


class TestSessionManagerProperties:
    """Test SessionManager properties."""

    def test_session_ids_returns_list(self):
        """Test session_ids returns list of IDs."""
        manager = SessionManager()
        manager.add("a", MockSession())
        manager.add("b", MockSession())
        manager.add("c", MockSession())

        ids = manager.session_ids

        assert isinstance(ids, list)
        assert set(ids) == {"a", "b", "c"}

    def test_len_returns_session_count(self):
        """Test len returns number of sessions."""
        manager = SessionManager()

        assert len(manager) == 0

        manager.add("s1", MockSession())
        assert len(manager) == 1

        manager.add("s2", MockSession())
        assert len(manager) == 2


class TestSessionManagerIntegration:
    """Integration tests for SessionManager."""

    def test_realistic_setup(self):
        """Test realistic session manager setup."""
        manager = SessionManager()

        # Add different types of sessions
        manager.add("default", MockSession("default"))
        manager.add("backup", MockSession("backup"))
        manager.add("lazy_special", MockSession("special"), lazy=True)

        assert len(manager) == 3
        assert manager.default_session_id == "default"
        assert "lazy_special" in manager._lazy_sessions

    @pytest.mark.asyncio
    async def test_lifecycle_management(self):
        """Test complete lifecycle of session manager."""
        manager = SessionManager()
        sessions = [MockSession(f"s{i}") for i in range(3)]

        for i, session in enumerate(sessions):
            manager.add(f"session{i}", session)

        # Before start - no sessions active
        assert all(not s._is_alive for s in sessions)

        # After start - all active
        await manager.start()
        assert all(s._is_alive for s in sessions)

        # After close - all inactive
        await manager.close()
        assert all(not s._is_alive for s in sessions)


class TestSessionManagerFetch:
    """Test SessionManager fetch behavior."""

    @pytest.mark.asyncio
    async def test_fetch_preserves_request_method(self):
        """Test that fetch does not mutate request._session_kwargs.

        Previously, fetch() used pop("method") which removed the method
        key from the original request dict. This caused retried requests
        (via request.copy()) to lose their HTTP method and fall back to GET.
        """
        from scrapling.engines.static import _ASyncSessionLogic
        from scrapling.fetchers import FetcherSession
        from scrapling.engines.toolbelt.custom import Response

        mock_response = Response(
            url="https://example.com",
            content=b"ok",
            status=200,
            reason="OK",
            cookies={},
            headers={"content-type": "text/html"},
            request_headers={},
        )
        mock_response.meta = {}

        mock_client = AsyncMock(spec=_ASyncSessionLogic)
        mock_client._make_request = AsyncMock(return_value=mock_response)

        mock_session = AsyncMock(spec=FetcherSession)
        mock_session._client = mock_client
        mock_session._is_alive = True

        manager = SessionManager()
        manager._sessions["default"] = mock_session
        manager._default_session_id = "default"
        manager._started = True

        request = Request("https://example.com", method="POST", data={"key": "value"})

        assert request._session_kwargs["method"] == "POST"

        await manager.fetch(request)

        # method must still be present after fetch
        assert "method" in request._session_kwargs
        assert request._session_kwargs["method"] == "POST"

        # verify the correct method was passed to _make_request
        mock_client._make_request.assert_called_once()
        call_kwargs = mock_client._make_request.call_args
        assert call_kwargs.kwargs["method"] == "POST"

    @staticmethod
    def _manager_with_mock_client():
        """Build a SessionManager whose default FetcherSession records `_make_request` calls."""
        from scrapling.engines.static import _ASyncSessionLogic
        from scrapling.fetchers import FetcherSession
        from scrapling.engines.toolbelt.custom import Response

        mock_response = Response(
            url="https://example.com",
            content=b"ok",
            status=200,
            reason="OK",
            cookies={},
            headers={"content-type": "text/html"},
            request_headers={},
        )
        mock_response.meta = {}

        mock_client = AsyncMock(spec=_ASyncSessionLogic)
        mock_client._make_request = AsyncMock(return_value=mock_response)

        mock_session = AsyncMock(spec=FetcherSession)
        mock_session._client = mock_client
        mock_session._is_alive = True

        manager = SessionManager()
        manager._sessions["default"] = mock_session
        manager._default_session_id = "default"
        manager._started = True
        return manager, mock_client

    @pytest.mark.asyncio
    async def test_fetch_passes_stealthy_headers_as_stealth(self):
        """`stealthy_headers` is mapped to `stealth` like the public session methods do, not sent to curl_cffi."""
        manager, mock_client = self._manager_with_mock_client()
        request = Request("https://example.com", stealthy_headers=False)

        await manager.fetch(request)

        call_kwargs = mock_client._make_request.call_args.kwargs
        assert call_kwargs["stealth"] is False
        assert "stealthy_headers" not in call_kwargs
        # The request keeps its kwargs, so retries and copies still carry the setting
        assert request._session_kwargs["stealthy_headers"] is False

    @pytest.mark.asyncio
    async def test_fetch_still_accepts_stealth(self):
        """Passing the internal `stealth` name directly keeps working."""
        manager, mock_client = self._manager_with_mock_client()

        await manager.fetch(Request("https://example.com", stealth=False))

        assert mock_client._make_request.call_args.kwargs["stealth"] is False

    @pytest.mark.asyncio
    async def test_fetch_without_stealthy_headers_uses_session_default(self):
        """Without either name, nothing is passed and the session falls back to its own setting."""
        manager, mock_client = self._manager_with_mock_client()

        await manager.fetch(Request("https://example.com"))

        call_kwargs = mock_client._make_request.call_args.kwargs
        assert "stealth" not in call_kwargs
        assert "stealthy_headers" not in call_kwargs


class LifecycleSession(MockSession):
    """Deterministic lifecycle failures without network or browser resources."""

    def __init__(self, name, events, *, enter_error=None, exit_error=None, alive=False):
        super().__init__(name)
        self.events = events
        self.enter_error = enter_error
        self.exit_error = exit_error
        self._is_alive = alive
        self.enter_hook = None
        self.exit_hook = None

    async def __aenter__(self):
        self.events.append(f"enter:{self.name}")
        if self.enter_hook:
            await self.enter_hook()
        if self.enter_error:
            raise self.enter_error
        return await super().__aenter__()

    async def __aexit__(self, *args):
        self.events.append(f"exit:{self.name}")
        if self.exit_hook:
            await self.exit_hook()
        if self.exit_error:
            raise self.exit_error
        if not self._is_alive:
            raise RuntimeError("double close")
        await super().__aexit__(*args)


class TestLifecycleSafety:
    @pytest.mark.asyncio
    async def test_partial_start_rolls_back_only_new_sessions(self):
        events = []
        manager = SessionManager()
        old = LifecycleSession("old", events, alive=True)
        a = LifecycleSession("a", events)
        b = LifecycleSession("b", events)
        error = ValueError("startup")
        bad = LifecycleSession("bad", events, enter_error=error)
        lazy = LifecycleSession("lazy", events)
        for session in (old, a, b, bad):
            manager.add(session.name, session)
        manager.add("lazy", lazy, lazy=True)
        with pytest.raises(ValueError) as caught:
            await manager.start()
        assert caught.value is error
        assert events == ["enter:a", "enter:b", "enter:bad", "exit:b", "exit:a"]
        assert old._is_alive and not a._is_alive and not b._is_alive
        assert not lazy._is_alive and not manager._started
        bad.enter_error = None
        await manager.start()
        await manager.close()
        assert not old._is_alive and not manager._started

    @pytest.mark.asyncio
    async def test_rollback_failure_preserves_startup_error_and_diagnostics(self, caplog):
        events = []
        manager = SessionManager()
        a = LifecycleSession("a", events)
        b = LifecycleSession("b", events, exit_error=RuntimeError("rollback failure"))
        original = ValueError("original startup")
        manager.add("a", a).add("b", b).add("bad", LifecycleSession("bad", events, enter_error=original))
        with pytest.raises(ValueError) as caught:
            await manager.start()
        assert caught.value is original
        assert events[-2:] == ["exit:b", "exit:a"]
        assert not a._is_alive and b._is_alive and not manager._started
        assert "rollback failure" in caplog.text
        assert any(record.exc_info for record in caplog.records)
        b.exit_error = None
        await manager.close()
        assert not b._is_alive

    @pytest.mark.asyncio
    @pytest.mark.parametrize("multiple", [False, True])
    async def test_close_attempts_all_and_resets_state(self, multiple, caplog):
        events = []
        manager = SessionManager()
        first = ValueError("first close")
        sessions = [
            LifecycleSession("a", events, exit_error=first),
            LifecycleSession("b", events, exit_error=RuntimeError("second close") if multiple else None),
            LifecycleSession("c", events),
        ]
        for session in sessions:
            manager.add(session.name, session)
        await manager.start()
        with pytest.raises(ValueError) as caught:
            await manager.close()
        assert caught.value is first
        assert events[-3:] == ["exit:a", "exit:b", "exit:c"]
        assert not sessions[-1]._is_alive and not manager._started
        if multiple:
            assert "second close" in caplog.text
        for session in sessions:
            session.exit_error = None
        await manager.close()
        assert all(not session._is_alive for session in sessions)

    @pytest.mark.asyncio
    async def test_normal_context_lazy_and_repeated_calls(self):
        from types import SimpleNamespace

        events = []
        manager = SessionManager()
        eager = LifecycleSession("eager", events)
        lazy = LifecycleSession("lazy", events)
        unused = LifecycleSession("unused", events)

        async def fetch(**kwargs):
            return SimpleNamespace(meta={})

        lazy.fetch = fetch
        manager.add("eager", eager).add("lazy", lazy, lazy=True).add("unused", unused, lazy=True)
        await manager.close()  # No session has been entered.
        async with manager:
            await manager.start()
            assert events == ["enter:eager"]
            await manager.fetch(Request("https://example.invalid/", sid="lazy"))
            assert lazy._is_alive and not unused._is_alive
        await manager.close()
        assert events == ["enter:eager", "enter:lazy", "exit:eager", "exit:lazy"]
        assert not manager._started
        await manager.start()
        await manager.close()
        assert events[-2:] == ["enter:eager", "exit:eager"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phase", ["start", "close"])
    async def test_asyncio_cancellation_attempts_remaining_cleanup(self, phase):
        import asyncio

        events = []
        manager = SessionManager()
        a = LifecycleSession("a", events)
        b = LifecycleSession("b", events)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def block():
            entered.set()
            await release.wait()

        manager.add("a", a).add("b", b)
        if phase == "start":
            b.enter_hook = block
        else:
            await manager.start()
            a.exit_hook = block
        task = asyncio.create_task(manager.start() if phase == "start" else manager.close())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if phase == "start":
            assert not manager._started
            assert not a._is_alive and "exit:a" in events
        else:
            # Shared-task model: cleanup continues in the background task
            # after the caller is cancelled.  Release the blocked session and
            # await the shared task to completion.
            release.set()
            await manager.close()
            assert not manager._started
            assert not a._is_alive and not b._is_alive
            assert events[-2:] == ["exit:a", "exit:b"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phase", ["start", "close"])
    async def test_anyio_cancel_scope_allows_cleanup_and_preserves_cancellation(self, phase):
        import anyio

        events = []
        manager = SessionManager()
        a = LifecycleSession("a", events)
        b = LifecycleSession("b", events)

        async def cleanup_checkpoint():
            await anyio.sleep(0)

        a.exit_hook = cleanup_checkpoint
        b.exit_hook = cleanup_checkpoint
        manager.add("a", a).add("b", b)
        if phase == "close":
            await manager.start()
        with anyio.CancelScope() as scope:

            async def cancel_start():
                scope.cancel()
                await anyio.sleep(0)

            if phase == "start":
                b.enter_hook = cancel_start
            else:
                scope.cancel()
            with pytest.raises(anyio.get_cancelled_exc_class()):
                await (manager.start() if phase == "start" else manager.close())
        if phase == "close":
            # Shared-task model: cleanup continues in the background task
            # after the caller is cancelled.  Await its completion outside
            # the cancelled scope.
            await manager.close()
        assert not manager._started and not a._is_alive
        if phase == "close":
            assert not b._is_alive and events[-2:] == ["exit:a", "exit:b"]

    @pytest.mark.asyncio
    async def test_successful_multi_session_lifecycle(self):
        events = []
        manager = SessionManager()
        sessions = [LifecycleSession(name, events) for name in ("a", "b", "c")]
        for session in sessions:
            manager.add(session.name, session)
        async with manager:
            await manager.start()
            assert manager._started and all(session._is_alive for session in sessions)
        await manager.close()
        assert events == ["enter:a", "enter:b", "enter:c", "exit:a", "exit:b", "exit:c"]
        assert not manager._started

    @pytest.mark.asyncio
    async def test_shutdown_cancellation_is_not_hidden_by_an_earlier_error(self, caplog):
        import asyncio

        events = []
        manager = SessionManager()
        cancellation = asyncio.CancelledError("cancel close")
        manager.add("a", LifecycleSession("a", events, exit_error=ValueError("first error")))
        manager.add("b", LifecycleSession("b", events, exit_error=cancellation))
        c = LifecycleSession("c", events)
        manager.add("c", c)
        await manager.start()
        with pytest.raises(asyncio.CancelledError) as caught:
            await manager.close()
        assert caught.value is cancellation
        assert not c._is_alive and not manager._started
        assert events[-3:] == ["exit:a", "exit:b", "exit:c"]
        assert "first error" in caplog.text

    @pytest.mark.asyncio
    async def test_anyio_cancellation_during_shutdown_is_delivered_after_cleanup(self):
        import anyio

        events = []
        manager = SessionManager()
        a = LifecycleSession("a", events)
        b = LifecycleSession("b", events)
        manager.add("a", a).add("b", b)
        await manager.start()
        with anyio.CancelScope() as scope:

            async def cancel_during_cleanup():
                scope.cancel()
                await anyio.sleep(0)

            a.exit_hook = cancel_during_cleanup
            with pytest.raises(anyio.get_cancelled_exc_class()):
                await manager.close()
        # Shared-task model: cleanup continues in the background task after
        # the caller is cancelled.  Await its completion outside the cancelled
        # scope so cancellation is delivered after cleanup finishes.
        await manager.close()
        assert events[-2:] == ["exit:a", "exit:b"]
        assert not a._is_alive and not b._is_alive and not manager._started

    @pytest.mark.asyncio
    async def test_cancelled_scope_is_not_masked_by_shutdown_error(self, caplog):
        import anyio

        events = []
        manager = SessionManager()
        a = LifecycleSession("a", events, exit_error=ValueError("close under cancellation"))
        b = LifecycleSession("b", events)
        manager.add("a", a).add("b", b)
        await manager.start()
        with anyio.CancelScope() as scope:
            scope.cancel()
            with pytest.raises(anyio.get_cancelled_exc_class()):
                await manager.close()
        # Shared-task model: cleanup continues in the background task after
        # the caller is cancelled.  The shared task ran session __aexit__ for
        # both sessions: session b closed successfully, session a failed with
        # ValueError (logged).  _close_done is False because cleanup failed,
        # so a second close() retries and the failure is observable again.
        with pytest.raises(ValueError, match="close under cancellation"):
            await manager.close()
        assert not b._is_alive and not manager._started
        assert "close under cancellation" in caplog.text
