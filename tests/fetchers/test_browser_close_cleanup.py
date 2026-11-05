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

from scrapling.engines._browsers._base import AsyncSession, SyncSession


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
