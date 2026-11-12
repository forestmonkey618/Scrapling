import asyncio
from asyncio import Lock

from anyio import CancelScope
from anyio.lowlevel import checkpoint_if_cancelled

from scrapling.core.utils import log

from scrapling.spiders.request import Request
from scrapling.engines.static import _ASyncSessionLogic
from scrapling.engines.toolbelt.convertor import Response
from scrapling.core._types import Set, cast, SUPPORTED_HTTP_METHODS
from scrapling.fetchers import AsyncStealthySession, FetcherSession

Session = FetcherSession | AsyncStealthySession


class SessionManager:
    """Manages pre-configured session instances."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._default_session_id: str | None = None
        self._started: bool = False
        self._lazy_sessions: Set[str] = set()
        self._lazy_lock = Lock()
        # Coordinates concurrent close() calls so the same session's __aexit__()
        # is never executed simultaneously.  A single close() holds the lock for
        # the entire duration of cleanup; concurrent callers block until it
        # releases the lock, then see _close_done and return.  _close_done is
        # only set when cleanup completes successfully, so an interrupted
        # close() (cancelled or errored) leaves _close_done False and a
        # subsequent close() can retry unfinished cleanup.
        self._close_lock = Lock()
        self._close_task: asyncio.Task | None = None
        self._close_done: bool = False
        self._close_error: BaseException | None = None

    def add(self, session_id: str, session: Session, *, default: bool = False, lazy: bool = False) -> "SessionManager":
        """Register a session instance.

        :param session_id: Name to reference this session in requests
        :param session: Your pre-configured session instance
        :param default: If True, this becomes the default session
        :param lazy: If True, the session will be started only when a request uses its ID.
        """
        if session_id in self._sessions:
            raise ValueError(f"Session '{session_id}' already registered")

        self._sessions[session_id] = session

        if default or self._default_session_id is None:
            self._default_session_id = session_id

        if lazy:
            self._lazy_sessions.add(session_id)

        return self

    def remove(self, session_id: str) -> None:
        """Removes a session.

        :param session_id: ID of session to remove
        """
        _ = self.pop(session_id)

    def pop(self, session_id: str) -> Session:
        """Remove and returns a session.

        :param session_id: ID of session to remove
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found")

        session = self._sessions.pop(session_id)
        if session_id in self._lazy_sessions:
            self._lazy_sessions.remove(session_id)

        if session and self._default_session_id == session_id:
            self._default_session_id = next(iter(self._sessions), None)

        return session

    @property
    def default_session_id(self) -> str:
        if self._default_session_id is None:
            raise RuntimeError("No sessions registered")
        return self._default_session_id

    @property
    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            available = ", ".join(self._sessions.keys())
            raise KeyError(f"Session '{session_id}' not found. Available: {available}")
        return self._sessions[session_id]

    async def start(self) -> None:
        """Start all sessions that aren't already alive.

        If a close() is in flight (_close_task is not None), wait for it to
        complete before evaluating _started.  _started stays True during close
        (only set to False in _do_close's finally), so a naive check would
        falsely report sessions as already started while they are being shut
        down.
        """
        # Wait for any in-flight close to finish before checking _started.
        async with self._close_lock:
            close_task = self._close_task
        if close_task is not None:
            await asyncio.shield(close_task)

        if self._started:
            return

        opened: list[Session] = []
        try:
            for sid, session in self._sessions.items():
                if sid not in self._lazy_sessions and not session._is_alive:
                    await session.__aenter__()
                    opened.append(session)
        except BaseException:
            # Only roll back sessions opened by this call, not pre-existing ones.
            with CancelScope(shield=True):
                for session in reversed(opened):
                    try:
                        await session.__aexit__(None, None, None)
                    except BaseException:
                        log.error("Session startup rollback failed", exc_info=True)
            raise

        self._started = True
        # A new start invalidates any previous close state so that a subsequent
        # close() actually performs cleanup instead of returning early.
        self._close_done = False
        self._close_task = None
        self._close_error = None

    async def close(self) -> None:
        """Attempt every applicable close, then propagate failure (cancellation takes priority).

        Concurrent close() calls are coordinated through a single shared cleanup
        task: only one caller executes session __aexit__() while others await the
        same task.  A close() after cleanup completes is a no-op (idempotent).
        Cleanup continues even if an individual caller is cancelled, so no
        resource-release operation is ever abandoned mid-flight.
        """
        async with self._close_lock:
            if self._close_done:
                return
            task = self._close_task

        # Another caller owns the cleanup task; await it with shielding so our
        # cancellation does not kill the shared cleanup.
        if task is not None:
            # Deliver any pending cancellation (e.g. from an anyio cancel
            # scope) before awaiting the shared task, so cancellation takes
            # priority even when the shield completes without suspending.
            await checkpoint_if_cancelled()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Caller cancelled before or while the shared task was running.
                # Cancellation takes priority over any task error.  The shared
                # task continues running (shielded) and its completion state
                # (_close_done, _close_error) is managed by the task itself.
                raise
            except BaseException as error:
                # The shared task completed with a non-cancellation error.
                # If our caller was also cancelled, cancellation takes priority.
                if asyncio.current_task().cancelling() > 0:
                    raise asyncio.CancelledError() from error
                raise
            # _do_close() stores errors in _close_error instead of raising,
            # so shield() completes normally even when cleanup failed.  Check
            # the stored error so concurrent callers observe the same failure.
            error = self._close_error
            if error is not None:
                raise error
            return

        # We own the cleanup task.
        async def _do_close() -> None:
            was_started = self._started
            first_error: BaseException | None = None
            try:
                # Let cleanup proceed even inside an already-cancelled AnyIO
                # scope, so an interrupted shutdown still completes.
                with CancelScope(shield=True):
                    for sid, session in self._sessions.items():
                        # Skip sessions that are not alive and either lazy or
                        # never started — they were never opened by this manager
                        # so there's nothing to close.  _is_alive is authoritative:
                        # after an interrupted __aexit__ it stays True (set only
                        # at the very end of close()), so unfinished cleanup is
                        # never skipped on a subsequent close().
                        if not session._is_alive and (sid in self._lazy_sessions or not was_started):
                            continue
                        # Each session's __aexit__ runs in its own shielded scope
                        # so that cancellation triggered by one session's exit hook
                        # (e.g. scope.cancel() in a test hook) does not prevent
                        # remaining sessions from being closed.
                        with CancelScope(shield=True):
                            try:
                                await session.__aexit__(None, None, None)
                            except BaseException as error:
                                log.error("Session shutdown failed", exc_info=True)
                                # Do NOT mark the session as closed: a failed
                                # __aexit__() does not establish that resources
                                # were released.  _is_alive stays True so a
                                # subsequent close() can retry cleanup for this
                                # session.  The error is recorded in _close_error
                                # and propagated by close().
                                if first_error is None or (
                                    isinstance(first_error, Exception) and not isinstance(error, Exception)
                                ):
                                    # Cancellation/system exceptions take priority
                                    # over ordinary failures.
                                    first_error = error
            finally:
                # Only the task itself marks completion, under the lock.
                # Note: we do NOT re-raise first_error here because asyncio
                # treats CancelledError raised inside a task as task
                # cancellation, which would lose the error instance.  The
                # error is stored in _close_error and propagated by close()
                # after awaiting the task.
                async with self._close_lock:
                    self._started = False
                    self._close_task = None
                    self._close_done = first_error is None
                    self._close_error = first_error

        async with self._close_lock:
            # Re-check under lock: another caller may have started cleanup while
            # we were outside the lock.
            if self._close_done:
                return
            if self._close_task is not None:
                task = self._close_task
            else:
                self._close_task = asyncio.create_task(_do_close())
                task = self._close_task

        # Deliver any pending cancellation before awaiting the shared task,
        # so cancellation takes priority even when the shield completes
        # without suspending.
        await checkpoint_if_cancelled()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Our caller was cancelled. Cancellation takes priority over any
            # task error: the shared task continues running (shielded) and its
            # error is preserved in _close_error for future callers.
            raise
        # Check cancellation again after the shield completes.  If the shield
        # finished without suspending (task already done), any pending
        # cancellation from an anyio cancel scope may not have been delivered
        # yet; this check ensures cancellation takes priority over task error.
        await checkpoint_if_cancelled()
        # Propagate any error the shared task stored (including CancelledError
        # raised by a session's __aexit__, which asyncio would otherwise
        # convert to task cancellation and lose the error instance).
        # Do NOT clear _close_error here: concurrent callers that awaited the
        # same task need to observe it too.  A subsequent successful close()
        # clears _close_error when it resets _close_error = None in its
        # finally block.
        error = self._close_error
        if error is not None:
            raise error

    async def fetch(self, request: Request) -> Response:
        sid = request.sid if request.sid else self.default_session_id
        session = self.get(sid)

        if session:
            if sid in self._lazy_sessions and not session._is_alive:
                async with self._lazy_lock:
                    if not session._is_alive:
                        await session.__aenter__()

            if isinstance(session, FetcherSession):
                client = session._client

                if isinstance(client, _ASyncSessionLogic):
                    kwargs = request._session_kwargs.copy()
                    method = cast(SUPPORTED_HTTP_METHODS, kwargs.pop("method", "GET"))
                    if "stealthy_headers" in kwargs:
                        # Same mapping as the public session methods (`get`, `post`, ...), so it doesn't reach curl_cffi
                        kwargs["stealth"] = kwargs.pop("stealthy_headers")
                    response = await client._make_request(
                        method=method,
                        url=request.url,
                        **kwargs,
                    )
                else:
                    # Sync session or other types - shouldn't happen in async context
                    raise TypeError(f"Session type {type(client)} not supported for async fetch")
            else:
                response = await session.fetch(url=request.url, **request._session_kwargs)

            response.request = request
            # Merge request meta into response meta (response meta takes priority)
            response.meta = {**request.meta, **response.meta}
            return response
        raise RuntimeError("No session found with the request session id")

    async def __aenter__(self) -> "SessionManager":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def __contains__(self, session_id: str) -> bool:
        """Check if a session ID is registered."""
        return session_id in self._sessions

    def __len__(self) -> int:
        """Number of registered sessions."""
        return len(self._sessions)
