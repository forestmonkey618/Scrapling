"""Regression tests for browser session ``close()`` cleanup behavior.

These tests use mocked browser/context/playwright objects. No real browser is
downloaded or launched. They verify that :meth:`SyncSession.close` and
:meth:`AsyncSession.close` release every Scrapling-owned resource even when an
individual close/stop fails or the asyncio task is cancelled partway through
cleanup.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from scrapling.engines._browsers._base import AsyncSession, SyncSession, _add_exception_note


def _make_session(asynchronous: bool = False):
    if asynchronous:
        session = AsyncSession(record_requests=True)
        context = Mock()
        browser = Mock()
        playwright = Mock()
        context.close = AsyncMock()
        browser.close = AsyncMock()
        playwright.stop = AsyncMock()
    else:
        session = SyncSession(record_requests=True)
        context = Mock()
        browser = Mock()
        playwright = Mock()
    session.context = context
    session.browser = browser
    session.playwright = playwright
    session._is_alive = True
    return session, context, browser, playwright


def _assert_fully_closed(session, context, browser, playwright):
    """Every Scrapling-owned resource released and session marked not alive."""
    assert session._is_alive is False
    assert session.context is None
    assert session.browser is None
    assert session.playwright is None
    context.close.assert_called_once()
    browser.close.assert_called_once()
    playwright.stop.assert_called_once()


class TestSyncSessionClose:
    def test_close_continues_after_context_close_raises(self):
        session, context, browser, playwright = _make_session(False)
        context.close.side_effect = RuntimeError("context already closed")

        with pytest.raises(RuntimeError, match="context already closed"):
            session.close()

        _assert_fully_closed(session, context, browser, playwright)

    def test_close_continues_after_browser_close_raises(self):
        session, context, browser, playwright = _make_session(False)
        browser.close.side_effect = RuntimeError("browser close failed")

        with pytest.raises(RuntimeError, match="browser close failed"):
            session.close()

        _assert_fully_closed(session, context, browser, playwright)

    def test_close_clears_state_after_multiple_close_failures(self):
        session, context, browser, playwright = _make_session(False)
        context.close.side_effect = RuntimeError("context close failed")
        browser.close.side_effect = RuntimeError("browser close failed")

        with pytest.raises(RuntimeError):
            session.close()

        assert session._is_alive is False
        assert session.context is None
        assert session.browser is None
        assert session.playwright is None
        context.close.assert_called_once()
        browser.close.assert_called_once()
        playwright.stop.assert_called_once()

    def test_close_reports_secondary_cleanup_failures(self):
        """Multiple cleanup failures: primary error propagates, secondary failures observable."""
        session, context, browser, playwright = _make_session(False)
        context.close.side_effect = RuntimeError("context close failed")
        browser.close.side_effect = RuntimeError("browser close failed")
        playwright.stop.side_effect = RuntimeError("playwright stop failed")

        with pytest.raises(RuntimeError, match="context close failed") as exc_info:
            session.close()

        _assert_fully_closed(session, context, browser, playwright)
        notes = getattr(exc_info.value, "__notes__", []) or []
        assert any("browser close failed" in note for note in notes)
        assert any("playwright stop failed" in note for note in notes)


class TestAsyncSessionClose:
    @pytest.mark.asyncio
    async def test_close_continues_after_context_close_raises(self):
        session, context, browser, playwright = _make_session(True)
        context.close.side_effect = RuntimeError("context already closed")

        with pytest.raises(RuntimeError, match="context already closed"):
            await session.close()

        _assert_fully_closed(session, context, browser, playwright)

    @pytest.mark.asyncio
    async def test_close_continues_after_browser_close_raises(self):
        session, context, browser, playwright = _make_session(True)
        browser.close.side_effect = RuntimeError("browser close failed")

        with pytest.raises(RuntimeError, match="browser close failed"):
            await session.close()

        _assert_fully_closed(session, context, browser, playwright)

    @pytest.mark.asyncio
    async def test_close_completes_under_context_cancellation(self):
        """Cancellation during context.close() must not skip browser/playwright cleanup."""
        session, context, browser, playwright = _make_session(True)
        blocker = asyncio.Event()
        started = asyncio.Event()

        async def hanging_close():
            started.set()
            await blocker.wait()

        context.close.side_effect = hanging_close

        task = asyncio.create_task(session.close())
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            blocker.set()

        _assert_fully_closed(session, context, browser, playwright)

    @pytest.mark.asyncio
    async def test_close_completes_under_browser_cancellation(self):
        """Cancellation during browser.close() must not skip playwright.stop()."""
        session, context, browser, playwright = _make_session(True)
        blocker = asyncio.Event()
        started = asyncio.Event()

        async def hanging_browser_close():
            started.set()
            await blocker.wait()

        browser.close.side_effect = hanging_browser_close

        task = asyncio.create_task(session.close())
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            blocker.set()

        _assert_fully_closed(session, context, browser, playwright)

    @pytest.mark.asyncio
    async def test_close_completes_under_pages_cancellation(self):
        """Cancellation during close_pages() must not skip context/browser/playwright cleanup."""
        session, context, browser, playwright = _make_session(True)
        blocker = asyncio.Event()
        started = asyncio.Event()

        async def hanging_page_close():
            started.set()
            await blocker.wait()

        hanging_page = Mock()
        hanging_page.close = AsyncMock(side_effect=hanging_page_close)
        session.page_pool.add_page(hanging_page)

        task = asyncio.create_task(session.close())
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            blocker.set()

        _assert_fully_closed(session, context, browser, playwright)

    @pytest.mark.asyncio
    async def test_close_completes_under_playwright_stop_cancellation(self):
        """Cancellation during playwright.stop() must still propagate after cleanup."""
        session, context, browser, playwright = _make_session(True)
        blocker = asyncio.Event()
        started = asyncio.Event()

        async def hanging_stop():
            started.set()
            await blocker.wait()

        playwright.stop.side_effect = hanging_stop

        task = asyncio.create_task(session.close())
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            blocker.set()

        _assert_fully_closed(session, context, browser, playwright)

    @pytest.mark.asyncio
    async def test_close_reports_secondary_cleanup_failures(self):
        """Multiple cleanup failures: primary error propagates, secondary failures observable."""
        session, context, browser, playwright = _make_session(True)
        context.close.side_effect = RuntimeError("context close failed")
        browser.close.side_effect = RuntimeError("browser close failed")
        playwright.stop.side_effect = RuntimeError("playwright stop failed")

        with pytest.raises(RuntimeError, match="context close failed") as exc_info:
            await session.close()

        _assert_fully_closed(session, context, browser, playwright)
        notes = getattr(exc_info.value, "__notes__", []) or []
        assert any("browser close failed" in note for note in notes)
        assert any("playwright stop failed" in note for note in notes)


class TestAddExceptionNote:
    """``_add_exception_note()`` must work where ``BaseException.add_note()`` does not exist.

    ``add_note()`` arrived in Python 3.11.  ``_WithoutAddNote`` behaves like a
    Python 3.10 exception on every interpreter, so the fallback is exercised
    by the whole test matrix and not by the Python 3.10 job alone.
    """

    class _WithoutAddNote(Exception):
        add_note = None

    def test_note_is_stored_when_add_note_is_missing(self):
        exc = self._WithoutAddNote("primary")

        _add_exception_note(exc, "first")
        _add_exception_note(exc, "second")

        assert exc.__notes__ == ["first", "second"]

    def test_existing_notes_are_kept(self):
        exc = self._WithoutAddNote("primary")
        exc.__notes__ = ["already here"]

        _add_exception_note(exc, "added")

        assert exc.__notes__ == ["already here", "added"]

    def test_add_note_is_used_when_the_exception_provides_it(self):
        calls = []

        class WithAddNote(Exception):
            def add_note(self, note):
                calls.append(note)

        _add_exception_note(WithAddNote("primary"), "secondary")

        assert calls == ["secondary"]

    def test_exception_refusing_attributes_is_left_untouched(self):
        """Attaching a note is best effort: it must never raise in place of the primary error."""

        class Frozen(self._WithoutAddNote):
            def __setattr__(self, name, value):
                raise AttributeError(name)

        exc = Frozen("primary")

        _add_exception_note(exc, "dropped")

        assert getattr(exc, "__notes__", None) is None
