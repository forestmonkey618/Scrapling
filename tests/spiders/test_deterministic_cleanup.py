"""Deterministic cleanup verification tests for SessionManager.close().

These tests verify shutdown behavior under direct asyncio cancellation,
concurrent close calls, retry safety, and error handling — without
relying on asyncio.sleep() timing to establish cleanup completion.
"""

import asyncio

import pytest

from scrapling.spiders.session import SessionManager


class TrackingSession:
    """Minimal session that tracks lifecycle state for deterministic testing."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._is_alive = False
        self._closed = False

    async def __aenter__(self):
        self._is_alive = True
        self._closed = False
        return self

    async def __aexit__(self, *args):
        self._is_alive = False
        self._closed = True


class BlockingSession(TrackingSession):
    """Session whose __aexit__ blocks on an Event, mirroring real session behavior.

    _is_alive stays True during blocked __aexit__ (set to False only after
    resource release completes), so an interrupted close() leaves the session
    appearing still-alive.
    """

    def __init__(self, name: str = "blocking"):
        super().__init__(name)
        self.close_event = asyncio.Event()
        self.aexit_started = asyncio.Event()
        self.aexit_completed = asyncio.Event()
        self.aexit_calls = 0

    async def __aexit__(self, *args):
        self._closed = True
        self.aexit_calls += 1
        self.aexit_started.set()
        await self.close_event.wait()
        self._is_alive = False
        self.aexit_completed.set()


class TestDirectAsyncioCancellation:
    """Verify direct asyncio cancellation reaches __aexit__ and cleanup recovers."""

    @pytest.mark.asyncio
    async def test_cancellation_reaches_blocked_aexit(self):
        """Caller cancellation does not interrupt __aexit__ (shared-task model).

        Cancellation propagates to the caller, but the shared cleanup task
        continues running shielded.  __aexit__ is called exactly once and
        cleanup completes after the blocked event is released, with no retry.
        """
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True
        manager._close_done = False

        caller_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

        # __aexit__ is blocked on close_event.wait()
        assert not session.close_event.is_set()
        assert session._is_alive is True
        assert session.aexit_calls == 1
        assert manager._started is True, "_started stays True while cleanup runs"
        assert manager._close_done is False, "_close_done False while cleanup in flight"

        # Save reference to shared cleanup task before caller cancel
        shared_task = manager._close_task

        # Cancel the close caller - CancelledError propagates, but shared task
        # continues running (shielded).  __aexit__ is NOT interrupted.
        caller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller_task

        # Cleanup still in flight: _started True, _close_done False
        assert manager._started is True, "_started True while cleanup task runs"
        assert manager._close_done is False, "_close_done False while cleanup in flight"
        assert session._is_alive is True, "__aexit__ blocked - _is_alive still True"
        assert session.aexit_calls == 1, "__aexit__ called once before caller cancel"
        assert shared_task is not None, "Shared cleanup task must exist"
        assert not shared_task.done(), "Cleanup task not done"

        # Release the blocked __aexit__ - shared task completes cleanup
        session.close_event.set()
        await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)

        # Cleanup completed despite caller cancellation (no retry needed)
        assert shared_task.done() and not shared_task.cancelled()
        assert session._is_alive is False, "Cleanup completed - _is_alive now False"
        assert manager._close_done is True, "_close_done True after task completion"
        assert session.aexit_calls == 1, "__aexit__ called exactly once (shared task owns it)"

    @pytest.mark.asyncio
    async def test_cancellation_propagates_to_caller(self):
        """Caller's CancelledError is not swallowed by close()."""
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        close_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

    @pytest.mark.asyncio
    async def test_no_pending_cleanup_task_after_cancellation(self):
        """No background cleanup task remains after cancellation.

        Direct execution means no orphaned task — the cancelled close() simply
        stops, and a subsequent close() performs cleanup inline.
        """
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        close_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert close_task.done()
        assert close_task.cancelled()

        # Second close() performs cleanup directly (no task to await)
        session.close_event.set()
        await manager.close()
        assert session._is_alive is False


class TestConcurrentClose:
    """Verify concurrent close() calls coordinate correctly."""

    @pytest.mark.asyncio
    async def test_concurrent_close_no_duplicate_aexit(self):
        """Two concurrent close() calls do not execute __aexit__ simultaneously."""
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        close1 = asyncio.create_task(manager.close())
        close2 = asyncio.create_task(manager.close())

        # Wait for first __aexit__ to enter blocked state
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

        # __aexit__ called exactly once — second caller blocked on lock
        assert session.aexit_calls == 1, (
            f"Concurrent close() must not re-enter __aexit__; got {session.aexit_calls}"
        )
        assert session._is_alive is True

        # Release cleanup — both callers complete
        session.close_event.set()
        await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)

        await asyncio.gather(close1, close2, return_exceptions=True)

        assert session._is_alive is False
        assert session._closed is True
        assert session.aexit_calls == 1, "__aexit__ called exactly once total"
        assert close1.done() and not close1.cancelled()
        assert close2.done() and not close2.cancelled()
        assert manager._close_done is True


class TestRetrySafety:
    """Verify retry safety after interrupted cleanup."""

    @pytest.mark.asyncio
    async def test_retry_after_interrupted_close(self):
        """Interrupted close() is recoverable via subsequent close()."""
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        # First close() gets interrupted
        close_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert session._is_alive is True, "Interrupted cleanup left session alive"
        assert manager._close_done is False, "_close_done False after interrupted close"
        assert manager._close_task is not None, "Shared cleanup task still running"

        # Second close() awaits the same in-flight task - no duplicate __aexit__
        session.close_event.set()
        await manager.close()
        assert session._is_alive is False, "Cleanup completed via shared task"
        assert manager._close_done is True, "_close_done True after successful close"
        assert session.aexit_calls == 1, "__aexit__ called exactly once (shared task owns it)"

    @pytest.mark.asyncio
    async def test_idempotent_close_after_completion(self):
        """Close after successful completion is a no-op (idempotent)."""
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        # First close completes
        session.close_event.set()  # Pre-release so __aexit__ doesn't block
        await manager.close()
        assert session._is_alive is False
        assert manager._close_done is True
        calls_after_first = session.aexit_calls

        # Second close() is idempotent — no-op
        await manager.close()
        assert session.aexit_calls == calls_after_first, (
            "Idempotent close() must not re-enter __aexit__"
        )
        assert manager._close_done is True


class TestErrorHandling:
    """Verify error handling in close()."""

    @pytest.mark.asyncio
    async def test_normal_close(self):
        """Normal close() completes all sessions and marks done."""
        manager = SessionManager()
        s1 = TrackingSession("s1")
        s2 = TrackingSession("s2")
        manager.add("s1", s1)
        manager.add("s2", s2)
        s1._is_alive = True
        s2._is_alive = True
        manager._started = True

        await manager.close()

        assert not s1._is_alive and not s2._is_alive
        assert s1._closed and s2._closed
        assert not manager._started
        assert manager._close_done is True

    @pytest.mark.asyncio
    async def test_cancelled_close_propagates(self):
        """Cancelled close() propagates CancelledError to caller."""
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        close_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)
        close_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await close_task

    @pytest.mark.asyncio
    async def test_runtimeerror_from_aexit(self):
        """RuntimeError from __aexit__ is observable after all sessions attempted."""
        manager = SessionManager()

        class FailingSession(TrackingSession):
            async def __aexit__(self, *args):
                self._is_alive = False
                self._closed = True
                raise RuntimeError("session close failed")

        s1 = FailingSession("failing")
        s2 = TrackingSession("ok")
        manager.add("s1", s1)
        manager.add("s2", s2)
        s1._is_alive = True
        s2._is_alive = True
        manager._started = True

        with pytest.raises(RuntimeError, match="session close failed"):
            await manager.close()

        # Both sessions received close attempts
        assert not s1._is_alive and not s2._is_alive
        assert s1._closed and s2._closed
        assert manager._close_done is False, "_close_done False after error"

    @pytest.mark.asyncio
    async def test_multiple_sessions_one_fails_others_still_closed(self):
        """When one session's __aexit__ fails, remaining sessions still close."""
        manager = SessionManager()

        class FailingSession(TrackingSession):
            async def __aexit__(self, *args):
                self._is_alive = False
                self._closed = True
                raise RuntimeError("s1 close failed")

        s1 = FailingSession("s1")
        s2 = TrackingSession("s2")
        s3 = TrackingSession("s3")
        manager.add("s1", s1)
        manager.add("s2", s2)
        manager.add("s3", s3)
        s1._is_alive = True
        s2._is_alive = True
        s3._is_alive = True
        manager._started = True

        with pytest.raises(RuntimeError, match="s1 close failed"):
            await manager.close()

        # All sessions received close attempts
        assert not s1._is_alive and not s2._is_alive and not s3._is_alive
        assert s1._closed and s2._closed and s3._closed

    @pytest.mark.asyncio
    async def test_repeated_close_after_completion_no_errors(self):
        """Repeated close() after successful completion does not raise."""
        manager = SessionManager()
        session = TrackingSession("default")
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        await manager.close()
        await manager.close()  # Should not raise
        await manager.close()  # Should not raise

        assert manager._close_done is True
        assert not session._is_alive

    @pytest.mark.asyncio
    async def test_close_without_start_is_noop(self):
        """Close() on never-started manager is idempotent no-op."""
        manager = SessionManager()
        session = TrackingSession("default")
        manager.add("default", session)
        # Never started: _is_alive False, _started False

        await manager.close()
        await manager.close()

        assert manager._close_done is True
        assert not session._closed  # __aexit__ was never called


class TestLazySessionBehavior:
    """Verify lazy-session behavior is preserved during close()."""

    @pytest.mark.asyncio
    async def test_lazy_session_never_started_is_skipped(self):
        """Lazy session that was never started is skipped during close()."""
        manager = SessionManager()
        session = TrackingSession("lazy")
        manager.add("lazy", session, lazy=True)
        manager._started = True  # Manager started but lazy session never entered

        await manager.close()

        assert not session._is_alive
        assert not session._closed  # __aexit__ never called for never-started lazy session
        assert manager._close_done is True

    @pytest.mark.asyncio
    async def test_lazy_session_that_was_started_is_closed(self):
        """Lazy session that was started (via fetch) is closed during close()."""
        manager = SessionManager()
        session = BlockingSession("lazy")
        manager.add("lazy", session, lazy=True)
        session._is_alive = True  # Simulate it was started via fetch
        manager._started = True

        session.close_event.set()  # Pre-release so __aexit__ doesn't block
        await manager.close()

        assert not session._is_alive
        assert session._closed
        assert manager._close_done is True


# ---------------------------------------------------------------------------
# Focused regression tests (Audit #19, items A–H)
# ---------------------------------------------------------------------------


class TestCancellationNoRetry:
    """Item A: Cancellation during blocked __aexit__ with no second close call.

    With the shared cleanup task, caller cancellation does NOT abandon cleanup.
    The shared task continues running (shielded from caller cancellation).
    _close_done is False while the task is in-flight; it becomes True when the
    task completes.  If no second close() occurs, cleanup still completes in
    the background task.
    """

    @pytest.mark.asyncio
    async def test_cancellation_no_retry_cleanup_continues(self):
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True
        manager._close_done = False

        caller_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

        # Save reference to shared cleanup task before caller cancel
        shared_task = manager._close_task

        # Caller is cancelled - but the shared task continues running
        caller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller_task

        # Shared cleanup task is still running (__aexit__ blocked on close_event)
        assert shared_task is not None, "Shared cleanup task must still be running"
        assert not shared_task.done(), "Cleanup task not done"
        assert session._is_alive is True, "__aexit__ blocked - _is_alive still True"
        assert manager._close_done is False, "_close_done False while cleanup in flight"
        assert session.aexit_calls == 1, "__aexit__ called once before caller cancel"

        # Release the blocked __aexit__ - the shared task completes cleanup
        session.close_event.set()
        await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)

        # Cleanup completed despite caller cancellation (no retry needed)
        assert shared_task.done() and not shared_task.cancelled()
        assert session._is_alive is False, "Cleanup completed - _is_alive now False"
        assert manager._close_done is True, "_close_done True after task completion"
        assert session.aexit_calls == 1, "__aexit__ called exactly once (shared task owns it)"


class TestCancellationWithRetry:
    """Item B: Cancellation followed by a second close.

    With the shared cleanup task, the second close() awaits the same in-flight
    task instead of retrying __aexit__().  The task continues running after the
    first caller is cancelled (shielded), and the second caller receives its
    result.  __aexit__ is called exactly once.
    """

    @pytest.mark.asyncio
    async def test_cancellation_then_second_close_awaits_task(self):
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True
        manager._close_done = False

        caller_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

        # First caller cancelled - shared cleanup task continues running (shielded)
        caller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller_task

        cleanup_task = manager._close_task
        assert cleanup_task is not None, "Shared cleanup task must still be running"
        assert not cleanup_task.done(), "Cleanup task not done"
        assert manager._close_done is False, "_close_done False while cleanup in flight"
        assert session._is_alive is True, "__aexit__ blocked - _is_alive still True"
        assert session.aexit_calls == 1, "__aexit__ called once so far"

        # Second close() awaits the same task - no duplicate __aexit__
        close_task2 = asyncio.create_task(manager.close())
        assert session.aexit_calls == 1, (
            "Second close() must await the same task, not re-enter __aexit__; "
            f"expected 1 call, got {session.aexit_calls}"
        )

        # Release cleanup - the shared task completes
        session.close_event.set()
        await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)
        await close_task2

        assert cleanup_task.done() and not cleanup_task.cancelled()
        assert close_task2.done() and not close_task2.cancelled()
        assert session._is_alive is False, "Cleanup completed - _is_alive now False"
        assert manager._close_done is True, "_close_done True after successful close"
        assert session.aexit_calls == 1, "__aexit__ called exactly once (shared task owns it)"


class TestConcurrentClose:
    """Item C: Two concurrent close callers.

    Verifies that only one __aexit__ executes and both callers terminate
    correctly.
    """

    @pytest.mark.asyncio
    async def test_two_concurrent_close_callers(self):
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        close1 = asyncio.create_task(manager.close())
        close2 = asyncio.create_task(manager.close())

        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)
        assert session.aexit_calls == 1, "Only one __aexit__ execution"

        session.close_event.set()
        await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)

        await asyncio.gather(close1, close2, return_exceptions=True)
        assert session._is_alive is False
        assert session._closed is True
        assert session.aexit_calls == 1
        assert close1.done() and not close1.cancelled()
        assert close2.done() and not close2.cancelled()
        assert manager._close_done is True


class TestCancellationAtCheckpoint:
    """Item D: Cancellation during close(), before checkpoint_if_cancelled().

    With the shared cleanup task, caller cancellation does not kill the task.
    _close_done is False while cleanup is in flight; it becomes True only when
    the shared task completes successfully.
    """

    @pytest.mark.asyncio
    async def test_cancellation_does_not_mark_done(self):
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True
        manager._close_done = False

        caller_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

        # Caller cancelled - shared task continues running (shielded)
        caller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller_task

        cleanup_task = manager._close_task
        assert cleanup_task is not None, "Shared cleanup task must still be running"
        assert not cleanup_task.done(), "Cleanup task not done"
        assert manager._close_done is False, "_close_done False while cleanup in flight"
        assert session._is_alive is True, "__aexit__ blocked - _is_alive still True"

        # Release - task completes, _close_done becomes True
        session.close_event.set()
        await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)
        assert cleanup_task.done() and not cleanup_task.cancelled()
        assert manager._close_done is True, "_close_done True after task completion"


class TestPartialCleanupFailureRetry:
    """Item E: Partial cleanup failure followed by retry.

    Verifies that when one session's __aexit__ raises, the error is
    observable, remaining sessions still close, and a subsequent close()
    retries the failed session.
    """

    @pytest.mark.asyncio
    async def test_partial_failure_then_retry(self):
        manager = SessionManager()

        class FailingThenOkSession(BlockingSession):
            def __init__(self):
                super().__init__("failing_then_ok")
                self.fail_first = True

            async def __aexit__(self, *args):
                self._closed = True
                self.aexit_calls += 1
                self.aexit_started.set()
                if self.fail_first:
                    self.fail_first = False
                    raise RuntimeError("first close failed")
                await self.close_event.wait()
                self._is_alive = False
                self.aexit_completed.set()

        s1 = FailingThenOkSession()
        s2 = TrackingSession("ok")
        manager.add("s1", s1)
        manager.add("s2", s2)
        s1._is_alive = True
        s2._is_alive = True
        manager._started = True

        # First close: s1 raises, s2 still closes
        with pytest.raises(RuntimeError, match="first close failed"):
            await manager.close()

        assert not s2._is_alive, "Remaining session still closed"
        assert s2._closed
        assert manager._close_done is False, "_close_done False after partial failure"

        # Retry: s1's __aexit__ no longer fails, cleanup completes
        s1.close_event.set()
        await manager.close()
        assert not s1._is_alive, "Retry completed cleanup for previously failing session"
        assert manager._close_done is True


class TestRepeatedStartCloseCycles:
    """Item F: Repeated start/close cycles.

    Verifies idempotency across multiple start/close cycles.
    """

    @pytest.mark.asyncio
    async def test_repeated_start_close_cycles(self):
        manager = SessionManager()
        session = TrackingSession("default")
        manager.add("default", session)

        for cycle in range(3):
            await manager.start()  # start() resets _close_done for a new cycle
            assert session._is_alive
            await manager.close()
            assert not session._is_alive
            assert manager._close_done is True
            # Second close in same cycle is idempotent
            await manager.close()
            assert manager._close_done is True


class TestResourceReleaseCompletion:
    """Item G: Resource-release completion independent of is_alive and close_done flags.

    Uses a session that tracks resource release separately from lifecycle
    flags to confirm actual completion, not just flag state.
    """

    @pytest.mark.asyncio
    async def test_resource_release_tracked_independently(self):
        manager = SessionManager()

        class ResourceTrackingSession(TrackingSession):
            def __init__(self):
                super().__init__("resource_tracked")
                self.resource_released = False

            async def __aexit__(self, *args):
                # Simulate resource release
                self.resource_released = True
                self._is_alive = False
                self._closed = True

        session = ResourceTrackingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True

        await manager.close()

        # Actual resource release observed, not just flag state
        assert session.resource_released is True, "Resource release did not complete"
        assert session._is_alive is False
        assert session._closed is True
        assert manager._close_done is True


class TestConsumerCancellationThroughStream:
    """Item H: Consumer cancellation through Spider.stream().

    Verifies that when a consumer cancels while iterating a stream,
    session cleanup completes and is not silently lost.  With the shared
    cleanup task, caller cancellation does not abandon cleanup - the shared
    task continues running and completes resource release.
    """

    @pytest.mark.asyncio
    async def test_consumer_cancellation_through_stream(self):
        manager = SessionManager()
        session = BlockingSession()
        manager.add("default", session)
        session._is_alive = True
        manager._started = True
        manager._close_done = False

        # Simulate consumer cancellation during close
        caller_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)
        caller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller_task

        # Shared cleanup task continues running (shielded from caller cancellation)
        cleanup_task = manager._close_task
        assert cleanup_task is not None, "Shared cleanup task must still be running"
        assert not cleanup_task.done(), "Cleanup task not done"
        assert manager._close_done is False, "_close_done False while cleanup in flight"
        assert session._is_alive is True, "__aexit__ blocked - _is_alive still True"

        # Release cleanup - the shared task completes without a retry
        session.close_event.set()
        await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)
        assert cleanup_task.done() and not cleanup_task.cancelled()
        assert session._is_alive is False, "Cleanup completed - _is_alive now False"
        assert manager._close_done is True, "_close_done True after task completion"


class TestConcurrentCloseFailurePropagation:
    """New: Concurrent close() callers must observe the same cleanup failure.

    When one session's __aexit__() raises during shared cleanup, every
    close() caller awaiting the shared task must receive the error — not
    just the owner.  The error is stored in _close_error and propagated
    to all waiters after the shared task completes.
    """

    @pytest.mark.asyncio
    async def test_concurrent_callers_observe_same_cleanup_failure(self):
        manager = SessionManager()

        class FailingSession:
            def __init__(self):
                self._is_alive = True
                self.exit_calls = 0

            async def __aenter__(self):
                self._is_alive = True
                return self

            async def __aexit__(self, *args):
                self.exit_calls += 1
                self._is_alive = False
                raise ValueError("cleanup failed")

        session = FailingSession()
        manager.add("default", session)
        manager._started = True

        # Two concurrent close() callers share the same cleanup task
        results: list[tuple[str, str]] = []

        async def close_and_capture(name: str) -> None:
            try:
                await manager.close()
                results.append((name, "SUCCESS - no error raised"))
            except ValueError as e:
                results.append((name, f"ValueError: {e}"))
            except BaseException as e:
                results.append((name, f"{type(e).__name__}: {e}"))

        c1 = asyncio.create_task(close_and_capture("caller-1"))
        c2 = asyncio.create_task(close_and_capture("caller-2"))
        await asyncio.gather(c1, c2)

        # Both callers must observe the cleanup failure
        errors = [r for _, r in results if "ValueError" in r]
        assert len(errors) == 2, (
            f"Both concurrent callers must observe cleanup failure; "
            f"got {len(errors)}/2: {results}"
        )
        assert session.exit_calls == 1, "__aexit__ called exactly once"


class TestCallerCancellationPriorityOverFailure:
    """New: Caller cancellation takes priority over a cleanup failure.

    When a caller is cancelled while the shared cleanup task fails with
    an error, the caller observes CancelledError (not the task error).
    The shared task's error is preserved in _close_error for future
    callers.
    """

    @pytest.mark.asyncio
    async def test_caller_cancellation_takes_priority_over_cleanup_failure(self):
        manager = SessionManager()

        class FailingSession:
            def __init__(self):
                self._is_alive = True

            async def __aenter__(self):
                self._is_alive = True
                return self

            async def __aexit__(self, *args):
                # A failed __aexit__() must not mark the session as closed
                # (_is_alive stays True) so a retry can re-attempt cleanup.
                raise ValueError("cleanup failed")

        session = FailingSession()
        manager.add("default", session)
        manager._started = True
        manager._close_done = False

        # First caller: cancelled while cleanup fails
        caller_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0)  # let close() start the shared task
        caller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller_task

        # Shared task failed and stored the error
        assert manager._close_error is not None, (
            "_close_error must store the cleanup failure for future callers"
        )
        assert isinstance(manager._close_error, ValueError)
        assert manager._close_done is False

        # Second caller: observes the stored error (not cancellation)
        with pytest.raises(ValueError, match="cleanup failed"):
            await manager.close()

        # _close_error persists after observation; only a successful close()
        # clears it (via _do_close's finally setting _close_error = None).
        assert manager._close_error is not None, (
            "_close_error persists after failed cleanup for future callers"
        )
        assert isinstance(manager._close_error, ValueError)
        assert manager._close_done is False  # cleanup failed, not done


class TestStartupDuringActiveShutdown:
    """New: start() must not falsely report success during active shutdown.

    When close() is in flight (_close_task is not None), _started stays
    True until _do_close's finally sets it to False (only on successful
    cleanup).  start() must wait for the in-flight close to complete before
    evaluating _started, so it doesn't return early while sessions are
    being shut down.
    """

    @pytest.mark.asyncio
    async def test_start_during_active_shutdown_waits_for_close(self):
        manager = SessionManager()

        class BlockingStartSession:
            def __init__(self):
                self._is_alive = False
                self.entered = False
                self.exited = False
                self.exit_started = asyncio.Event()
                self.release_exit = asyncio.Event()

            async def __aenter__(self):
                self.entered = True
                self._is_alive = True
                self.exited = False
                return self

            async def __aexit__(self, *args):
                self.exited = True
                self._is_alive = False
                self.exit_started.set()  # signal __aexit__ has started
                await self.release_exit.wait()  # block until test releases

        session = BlockingStartSession()
        manager.add("default", session)

        # Step 1: Start the session
        await manager.start()
        assert manager._started is True
        assert session._is_alive is True

        # Step 2: Begin close() — blocks inside __aexit__
        close_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(session.exit_started.wait(), timeout=5.0)
        assert manager._started is True, "_started True while close in flight"
        assert manager._close_task is not None, "Shared close task active"

        # Step 3: start() during active shutdown — must wait for close,
        # not return early with _started=True
        start_task = asyncio.create_task(manager.start())
        await asyncio.sleep(0)
        assert not start_task.done(), (
            "start() must wait for in-flight close, not return immediately"
        )

        # Step 4: Release the blocked __aexit__ — shared task completes
        session.release_exit.set()

        # start() should now proceed after close completes
        await asyncio.wait_for(start_task, timeout=5.0)
        assert not start_task.cancelled(), "start() completed after close finished"

        # The key verification: start() waited for close to complete before
        # proceeding (it did not return early while close was in flight).
        # After close completes, start() may restart the session, which is
        # correct lifecycle behavior.


class TestFailedCleanupFollowedByRetry:
    """Item E variant: Failed cleanup is observable and retryable.

    When cleanup fails, _close_done stays False and _close_error preserves
    the failure.  A subsequent close() retries cleanup and can succeed.
    """

    @pytest.mark.asyncio
    async def test_failed_cleanup_then_successful_retry(self):
        manager = SessionManager()

        class RetrySession:
            def __init__(self):
                self._is_alive = True
                self.exit_calls = 0
                self.fail_first = True

            async def __aenter__(self):
                self._is_alive = True
                return self

            async def __aexit__(self, *args):
                self.exit_calls += 1
                if self.fail_first:
                    self.fail_first = False
                    raise ValueError("first cleanup failed")
                self._is_alive = False

        session = RetrySession()
        manager.add("default", session)
        manager._started = True

        # First close: failure is observable
        with pytest.raises(ValueError, match="first cleanup failed"):
            await manager.close()
        assert manager._close_done is False
        assert manager._close_error is not None
        assert session.exit_calls == 1

        # Second close: retry succeeds
        await manager.close()
        assert manager._close_done is True
        assert manager._close_error is None
        assert session.exit_calls == 2, "__aexit__ retried on second close()"
        assert not session._is_alive, "Session closed after successful retry"