"""Tests for the CrawlerEngine class."""

import asyncio
import logging
import tempfile
from pathlib import Path

import anyio
import pytest

from scrapling.spiders.engine import CrawlerEngine, _dump
from scrapling.spiders.request import Request
from scrapling.spiders.spider import Spider
from scrapling.spiders.robotstxt import RobotsTxtManager
from scrapling.spiders.session import SessionManager
from scrapling.spiders.scheduler import Scheduler
from scrapling.spiders.result import CrawlStats, ItemList
from scrapling.spiders.checkpoint import CheckpointData
from scrapling.core._types import Any, Dict, Set, AsyncGenerator


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """Minimal Response stand-in."""

    def __init__(
        self, status: int = 200, body: bytes = b"ok", url: str = "https://example.com", encoding: str = "utf-8"
    ):
        self.status = status
        self.body = body
        self.url = url
        self.encoding = encoding
        self.request: Any = None
        self.meta: Dict[str, Any] = {}

    def __str__(self) -> str:
        return self.url


class MockSession:
    """Mock session that returns a canned response."""

    def __init__(self, name: str = "mock", response: MockResponse | None = None):
        self.name = name
        self._is_alive = False
        self._response = response or MockResponse()
        self.fetch_calls: list[dict] = []

    async def __aenter__(self):
        self._is_alive = True
        return self

    async def __aexit__(self, *args):
        self._is_alive = False

    async def fetch(self, url: str, **kwargs):
        self.fetch_calls.append({"url": url, **kwargs})
        resp = MockResponse(status=self._response.status, body=self._response.body, url=url)
        return resp


class ErrorSession(MockSession):
    """Session that raises on fetch."""

    def __init__(self, error: Exception | None = None):
        super().__init__("error")
        self._error = error or RuntimeError("fetch failed")

    async def fetch(self, url: str, **kwargs):
        raise self._error


class TrackingSession(MockSession):
    """Session that tracks whether it was closed."""

    def __init__(self, name: str = "mock", response: MockResponse | None = None):
        super().__init__(name, response)
        self.closed = False

    async def __aexit__(self, *args):
        self._is_alive = False
        self.closed = True


class MockSpider:
    """Lightweight spider stub for engine tests."""

    def __init__(
        self,
        *,
        concurrent_requests: int = 4,
        concurrent_requests_per_domain: int = 0,
        download_delay: float = 0.0,
        max_blocked_retries: int = 3,
        autothrottle_enabled: bool = False,
        autothrottle_start_delay: float = 5.0,
        autothrottle_max_delay: float = 60.0,
        autothrottle_target_concurrency: float | None = None,
        autothrottle_block_backoff: bool = True,
        allowed_domains: Set[str] | None = None,
        fp_include_kwargs: bool = False,
        fp_include_headers: bool = False,
        fp_keep_fragments: bool = False,
        is_blocked_fn=None,
        on_scraped_item_fn=None,
        retry_blocked_request_fn=None,
        robots_txt_obey: bool = False,
        start_urls: list[str] | None = None,
    ):
        self.concurrent_requests = concurrent_requests
        self.concurrent_requests_per_domain = concurrent_requests_per_domain
        self.download_delay = download_delay
        self.max_blocked_retries = max_blocked_retries
        self.autothrottle_enabled = autothrottle_enabled
        self.autothrottle_start_delay = autothrottle_start_delay
        self.autothrottle_max_delay = autothrottle_max_delay
        self.autothrottle_target_concurrency = autothrottle_target_concurrency
        self.autothrottle_block_backoff = autothrottle_block_backoff
        self.allowed_domains = allowed_domains or set()
        self.fp_include_kwargs = fp_include_kwargs
        self.fp_include_headers = fp_include_headers
        self.fp_keep_fragments = fp_keep_fragments
        self.name = "test_spider"
        self.robots_txt_obey = robots_txt_obey
        self.development_mode = False
        self.development_cache_dir = None
        self.start_urls = start_urls or []

        # Tracking lists
        self.on_start_calls: list[dict] = []
        self.on_close_calls: int = 0
        self.on_error_calls: list[tuple[Request, Exception]] = []
        self.scraped_items: list[dict] = []
        self.blocked_responses: list = []
        self.retry_requests: list = []

        # Pluggable behaviour
        self._is_blocked_fn = is_blocked_fn
        self._on_scraped_item_fn = on_scraped_item_fn
        self._retry_blocked_request_fn = retry_blocked_request_fn

        # Log counter stub
        self._log_counter = _LogCounterStub()

    async def parse(self, response) -> AsyncGenerator[Dict[str, Any] | Request | None, None]:
        yield {"url": str(response)}

    async def on_start(self, resuming: bool = False) -> None:
        self.on_start_calls.append({"resuming": resuming})

    async def on_close(self) -> None:
        self.on_close_calls += 1

    async def on_error(self, request: Request, error: Exception) -> None:
        self.on_error_calls.append((request, error))

    async def on_scraped_item(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        if self._on_scraped_item_fn:
            return self._on_scraped_item_fn(item)
        self.scraped_items.append(item)
        return item

    async def is_blocked(self, response) -> bool:
        if self._is_blocked_fn:
            return self._is_blocked_fn(response)
        return False

    async def retry_blocked_request(self, request: Request, response) -> Request:
        self.retry_requests.append(request)
        if self._retry_blocked_request_fn:
            return self._retry_blocked_request_fn(request, response)
        return request

    async def start_requests(self) -> AsyncGenerator[Request, None]:
        yield Request("https://example.com", sid="default")


class _LogCounterStub:
    """Stub for LogCounterHandler."""

    def get_counts(self) -> Dict[str, int]:
        return {"debug": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}


def _make_engine(
    spider: MockSpider | None = None,
    session: MockSession | None = None,
    crawldir: str | None = None,
    interval: float = 300.0,
) -> CrawlerEngine:
    """Create a CrawlerEngine wired to mock objects."""
    spider = spider or MockSpider()
    sm = SessionManager()
    sm.add("default", session or MockSession())
    return CrawlerEngine(spider, sm, crawldir=crawldir, interval=interval)


# ---------------------------------------------------------------------------
# Tests: _dump helper
# ---------------------------------------------------------------------------


class TestDumpHelper:
    def test_dump_returns_json_string(self):
        result = _dump({"key": "value"})
        assert '"key": "value"' in result

    def test_dump_handles_nested(self):
        result = _dump({"a": {"b": 1}})
        assert '"a"' in result
        assert '"b"' in result


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------


class TestCrawlerEngineInit:
    def test_default_initialisation(self):
        engine = _make_engine()

        assert engine._running is False
        assert engine._active_tasks == 0
        assert engine._pause_requested is False
        assert engine._force_stop is False
        assert engine.paused is False
        assert isinstance(engine.stats, CrawlStats)
        assert isinstance(engine.items, ItemList)

    def test_checkpoint_system_disabled_by_default(self):
        engine = _make_engine()
        assert engine._checkpoint_system_enabled is False

    def test_checkpoint_system_enabled_with_crawldir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(crawldir=tmpdir)
            assert engine._checkpoint_system_enabled is True

    def test_global_limiter_uses_concurrent_requests(self):
        spider = MockSpider(concurrent_requests=8)
        engine = _make_engine(spider=spider)
        assert engine._global_limiter.total_tokens == 8

    def test_allowed_domains_from_spider(self):
        spider = MockSpider(allowed_domains={"example.com", "test.org"})
        engine = _make_engine(spider=spider)
        assert engine._allowed_domains == {"example.com", "test.org"}


# ---------------------------------------------------------------------------
# Tests: _is_domain_allowed
# ---------------------------------------------------------------------------


class TestIsDomainAllowed:
    def test_all_allowed_when_empty(self):
        engine = _make_engine()
        request = Request("https://anything.com/page")
        assert engine._is_domain_allowed(request) is True

    def test_exact_domain_match(self):
        spider = MockSpider(allowed_domains={"example.com"})
        engine = _make_engine(spider=spider)

        assert engine._is_domain_allowed(Request("https://example.com/page")) is True
        assert engine._is_domain_allowed(Request("https://other.com/page")) is False

    def test_subdomain_match(self):
        spider = MockSpider(allowed_domains={"example.com"})
        engine = _make_engine(spider=spider)

        assert engine._is_domain_allowed(Request("https://sub.example.com/page")) is True
        assert engine._is_domain_allowed(Request("https://deep.sub.example.com/x")) is True

    def test_partial_name_not_matched(self):
        spider = MockSpider(allowed_domains={"example.com"})
        engine = _make_engine(spider=spider)

        # "notexample.com" should NOT match "example.com"
        assert engine._is_domain_allowed(Request("https://notexample.com/x")) is False

    def test_multiple_allowed_domains(self):
        spider = MockSpider(allowed_domains={"a.com", "b.org"})
        engine = _make_engine(spider=spider)

        assert engine._is_domain_allowed(Request("https://a.com/")) is True
        assert engine._is_domain_allowed(Request("https://b.org/")) is True
        assert engine._is_domain_allowed(Request("https://c.net/")) is False


# ---------------------------------------------------------------------------
# Tests: _rate_limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_returns_global_limiter_when_per_domain_disabled(self):
        engine = _make_engine()  # concurrent_requests_per_domain=0
        limiter = engine._rate_limiter("example.com")
        assert limiter is engine._global_limiter

    def test_returns_per_domain_limiter_when_enabled(self):
        spider = MockSpider(concurrent_requests_per_domain=2)
        engine = _make_engine(spider=spider)

        limiter = engine._rate_limiter("example.com")
        assert limiter is not engine._global_limiter
        assert limiter.total_tokens == 2

    def test_same_domain_returns_same_limiter(self):
        spider = MockSpider(concurrent_requests_per_domain=2)
        engine = _make_engine(spider=spider)

        l1 = engine._rate_limiter("example.com")
        l2 = engine._rate_limiter("example.com")
        assert l1 is l2

    def test_different_domains_get_different_limiters(self):
        spider = MockSpider(concurrent_requests_per_domain=2)
        engine = _make_engine(spider=spider)

        l1 = engine._rate_limiter("a.com")
        l2 = engine._rate_limiter("b.com")
        assert l1 is not l2


# ---------------------------------------------------------------------------
# Tests: _normalize_request
# ---------------------------------------------------------------------------


class TestNormalizeRequest:
    def test_sets_default_sid_when_empty(self):
        engine = _make_engine()
        request = Request("https://example.com")
        assert request.sid == ""

        engine._normalize_request(request)
        assert request.sid == "default"

    def test_preserves_existing_sid(self):
        engine = _make_engine()
        request = Request("https://example.com", sid="custom")

        engine._normalize_request(request)
        assert request.sid == "custom"


# ---------------------------------------------------------------------------
# Tests: _process_request
# ---------------------------------------------------------------------------


class TestProcessRequest:
    @pytest.mark.asyncio
    async def test_successful_fetch_updates_stats(self):
        spider = MockSpider()
        session = MockSession(response=MockResponse(status=200, body=b"hello"))
        engine = _make_engine(spider=spider, session=session)

        request = Request("https://example.com", sid="default")
        await engine._process_request(request)

        assert engine.stats.requests_count == 1
        assert engine.stats.response_bytes == 5  # len(b"hello") from MockSession
        assert "status_200" in engine.stats.response_status_count

    @pytest.mark.asyncio
    async def test_failed_fetch_increments_failed_count(self):
        spider = MockSpider()
        sm = SessionManager()
        sm.add("default", ErrorSession())
        engine = CrawlerEngine(spider, sm)

        request = Request("https://example.com", sid="default")
        await engine._process_request(request)

        assert engine.stats.failed_requests_count == 1
        assert len(spider.on_error_calls) == 1

    @pytest.mark.asyncio
    async def test_failed_fetch_does_not_increment_requests_count(self):
        spider = MockSpider()
        sm = SessionManager()
        sm.add("default", ErrorSession())
        engine = CrawlerEngine(spider, sm)

        request = Request("https://example.com", sid="default")
        await engine._process_request(request)

        assert engine.stats.requests_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", ["bfs", "dfs"])
    async def test_blocked_response_triggers_retry(self, strategy):
        spider = MockSpider(is_blocked_fn=lambda r: True, max_blocked_retries=2)
        engine = _make_engine(spider=spider)
        engine.scheduler = Scheduler(crawl_strategy=strategy)

        other = Request("https://example.com/other", sid="default")
        await engine.scheduler.enqueue(other)
        request = Request("https://example.com", sid="default")
        await engine._process_request(request)

        assert engine.stats.blocked_requests_count == 1
        # A retry request should be enqueued
        assert not engine.scheduler.is_empty
        assert await engine.scheduler.dequeue() is other
        retry = await engine.scheduler.dequeue()
        assert retry.url == request.url
        assert retry.priority == -1
        assert request.priority == 0
        assert retry._retry_count == 1
        assert retry.dont_filter

    @pytest.mark.asyncio
    async def test_blocked_response_max_retries_exceeded(self):
        spider = MockSpider(is_blocked_fn=lambda r: True, max_blocked_retries=2)
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default")
        request._retry_count = 2  # Already at max
        await engine._process_request(request)

        assert engine.stats.blocked_requests_count == 1
        # No retry enqueued
        assert engine.scheduler.is_empty

    @pytest.mark.asyncio
    async def test_retry_request_has_dont_filter(self):
        spider = MockSpider(is_blocked_fn=lambda r: True, max_blocked_retries=3)
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default")
        await engine._process_request(request)

        retry = await engine.scheduler.dequeue()
        assert retry.dont_filter is True
        assert retry._retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_clears_proxy_kwargs(self):
        spider = MockSpider(is_blocked_fn=lambda r: True, max_blocked_retries=3)
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default", proxy="http://proxy:8080")
        await engine._process_request(request)

        retry = await engine.scheduler.dequeue()
        assert "proxy" not in retry._session_kwargs
        assert "proxies" not in retry._session_kwargs

    @pytest.mark.asyncio
    async def test_callback_yielding_dict_increments_items(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default")
        await engine._process_request(request)

        assert engine.stats.items_scraped == 1
        assert len(engine.items) == 1

    @pytest.mark.asyncio
    async def test_callback_yielding_request_enqueues(self):
        async def callback(response) -> AsyncGenerator:
            yield Request("https://example.com/page2", sid="default")

        spider = MockSpider()
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default", callback=callback)
        await engine._process_request(request)

        assert not engine.scheduler.is_empty

    @pytest.mark.asyncio
    async def test_callback_yielding_offsite_request_filtered(self):
        async def callback(response) -> AsyncGenerator:
            yield Request("https://other.com/page", sid="default")

        spider = MockSpider(allowed_domains={"example.com"})
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default", callback=callback)
        await engine._process_request(request)

        assert engine.stats.offsite_requests_count == 1
        assert engine.scheduler.is_empty

    @pytest.mark.asyncio
    async def test_dropped_item_when_on_scraped_item_returns_none(self):
        spider = MockSpider(on_scraped_item_fn=lambda item: None)
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default")
        await engine._process_request(request)

        assert engine.stats.items_dropped == 1
        assert engine.stats.items_scraped == 0
        assert len(engine.items) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_calls_on_error(self):
        async def bad_callback(response) -> AsyncGenerator:
            raise ValueError("callback boom")
            yield  # noqa: unreachable

        spider = MockSpider()
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default", callback=bad_callback)
        await engine._process_request(request)

        assert len(spider.on_error_calls) == 1
        assert isinstance(spider.on_error_calls[0][1], ValueError)

    @pytest.mark.asyncio
    async def test_proxy_tracked_in_stats(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default", proxy="http://p:8080")
        await engine._process_request(request)

        assert "http://p:8080" in engine.stats.proxies

    @pytest.mark.asyncio
    async def test_proxies_dict_tracked_in_stats(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        proxies = {"http": "http://p:8080", "https": "https://p:8443"}
        request = Request("https://example.com", sid="default", proxies=proxies)
        await engine._process_request(request)

        assert len(engine.stats.proxies) == 1
        assert engine.stats.proxies[0] == proxies

    @pytest.mark.asyncio
    async def test_uses_parse_when_no_callback(self):
        items_seen = []

        async def custom_parse(response) -> AsyncGenerator:
            yield {"from": "custom_parse"}

        spider = MockSpider()
        spider.parse = custom_parse  # type: ignore[assignment]
        engine = _make_engine(spider=spider)

        request = Request("https://example.com", sid="default")
        # No callback set → should use spider.parse
        await engine._process_request(request)

        assert engine.stats.items_scraped == 1


# ---------------------------------------------------------------------------
# Tests: _task_wrapper
# ---------------------------------------------------------------------------


class TestTaskWrapper:
    @pytest.mark.asyncio
    async def test_decrements_active_tasks(self):
        engine = _make_engine()
        engine._active_tasks = 1

        request = Request("https://example.com", sid="default")
        await engine._task_wrapper(request)

        assert engine._active_tasks == 0

    @pytest.mark.asyncio
    async def test_decrements_even_on_error(self):
        spider = MockSpider()
        sm = SessionManager()
        sm.add("default", ErrorSession())
        engine = CrawlerEngine(spider, sm)
        engine._active_tasks = 1

        request = Request("https://example.com", sid="default")
        await engine._task_wrapper(request)

        assert engine._active_tasks == 0


# ---------------------------------------------------------------------------
# Tests: request_pause
# ---------------------------------------------------------------------------


class TestRequestPause:
    def test_first_call_sets_pause_requested(self):
        engine = _make_engine()

        engine.request_pause()

        assert engine._pause_requested is True
        assert engine._force_stop is False

    def test_second_call_sets_force_stop(self):
        engine = _make_engine()

        engine.request_pause()  # first
        engine.request_pause()  # second

        assert engine._pause_requested is True
        assert engine._force_stop is True

    def test_third_call_after_force_stop_is_noop(self):
        engine = _make_engine()

        engine.request_pause()
        engine.request_pause()
        engine.request_pause()  # should not raise

        assert engine._force_stop is True


# ---------------------------------------------------------------------------
# Tests: checkpoint methods
# ---------------------------------------------------------------------------


class TestCheckpointMethods:
    def test_is_checkpoint_time_false_when_disabled(self):
        engine = _make_engine()  # no crawldir
        assert engine._is_checkpoint_time() is False

    @pytest.mark.asyncio
    async def test_save_and_restore_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            engine = _make_engine(spider=spider, crawldir=tmpdir)

            # Enqueue a request so snapshot has data
            req = Request("https://example.com", sid="default")
            engine._normalize_request(req)
            await engine.scheduler.enqueue(req)

            await engine._save_checkpoint()

            # Verify checkpoint file exists
            checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
            assert checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_restore_when_no_checkpoint_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = _make_engine(crawldir=tmpdir)
            result = await engine._restore_from_checkpoint()
            assert result is False

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_raises_when_disabled(self):
        engine = _make_engine()  # no crawldir → checkpoint disabled
        assert (await engine._restore_from_checkpoint()) is False


# ---------------------------------------------------------------------------
# Tests: crawl
# ---------------------------------------------------------------------------


class TestCrawl:
    @pytest.mark.asyncio
    async def test_basic_crawl_returns_stats(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert isinstance(stats, CrawlStats)
        assert stats.requests_count >= 1
        assert stats.items_scraped >= 1

    @pytest.mark.asyncio
    async def test_crawl_calls_on_start_and_on_close(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        await engine.crawl()

        assert len(spider.on_start_calls) == 1
        assert spider.on_start_calls[0]["resuming"] is False
        assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_crawl_sets_stats_timing(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert stats.start_time > 0
        assert stats.end_time > 0
        assert stats.end_time >= stats.start_time

    @pytest.mark.asyncio
    async def test_crawl_sets_concurrency_stats(self):
        spider = MockSpider(concurrent_requests=16, concurrent_requests_per_domain=4)
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert stats.concurrent_requests == 16
        assert stats.concurrent_requests_per_domain == 4

    @pytest.mark.asyncio
    async def test_crawl_processes_multiple_start_urls(self):
        spider = MockSpider()

        urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]

        async def multi_start_requests() -> AsyncGenerator[Request, None]:
            for url in urls:
                yield Request(url, sid="default")

        spider.start_requests = multi_start_requests  # type: ignore[assignment]
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert stats.requests_count == 3
        assert stats.items_scraped == 3

    @pytest.mark.asyncio
    async def test_crawl_follows_yielded_requests(self):
        """Test that requests yielded from callbacks are processed."""
        call_count = 0

        async def parse_with_follow(response) -> AsyncGenerator:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Request("https://example.com/page2", sid="default")
            yield {"page": str(response)}

        spider = MockSpider()
        spider.parse = parse_with_follow  # type: ignore[assignment]
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert stats.requests_count == 2
        assert stats.items_scraped == 2

    @pytest.mark.asyncio
    async def test_crawl_with_download_delay(self):
        spider = MockSpider(download_delay=0.01)
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert stats.download_delay == 0.01
        assert stats.requests_count >= 1

    @pytest.mark.asyncio
    async def test_crawl_filters_offsite_requests(self):
        async def parse_offsite(response) -> AsyncGenerator:
            yield Request("https://other-domain.com/page", sid="default")
            yield {"url": str(response)}

        spider = MockSpider(allowed_domains={"example.com"})
        spider.parse = parse_offsite  # type: ignore[assignment]
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert stats.offsite_requests_count == 1
        assert stats.requests_count == 1  # Only the initial request

    @pytest.mark.asyncio
    async def test_crawl_cleans_up_checkpoint_on_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            engine = _make_engine(spider=spider, crawldir=tmpdir)

            await engine.crawl()

            checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
            assert not checkpoint_path.exists()  # Cleaned up

    @pytest.mark.asyncio
    async def test_crawl_handles_fetch_error_gracefully(self):
        spider = MockSpider()
        sm = SessionManager()
        sm.add("default", ErrorSession())
        engine = CrawlerEngine(spider, sm)

        stats = await engine.crawl()

        assert stats.failed_requests_count == 1
        assert len(spider.on_error_calls) == 1

    @pytest.mark.asyncio
    async def test_crawl_log_levels_populated(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        stats = await engine.crawl()

        assert isinstance(stats.log_levels_counter, dict)

    @pytest.mark.asyncio
    async def test_crawl_resets_state_on_each_run(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        # Run first crawl
        await engine.crawl()
        assert engine.stats.requests_count >= 1

        # Run second crawl - stats should reset
        stats = await engine.crawl()
        # Items are cleared on each crawl
        assert engine.paused is False


# ---------------------------------------------------------------------------
# Tests: items property
# ---------------------------------------------------------------------------


class TestItemsProperty:
    def test_items_returns_item_list(self):
        engine = _make_engine()
        assert isinstance(engine.items, ItemList)

    def test_items_initially_empty(self):
        engine = _make_engine()
        assert len(engine.items) == 0

    @pytest.mark.asyncio
    async def test_items_populated_after_crawl(self):
        engine = _make_engine()
        await engine.crawl()
        assert len(engine.items) >= 1


# ---------------------------------------------------------------------------
# Tests: streaming (__aiter__ / _stream)
# ---------------------------------------------------------------------------


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_yields_items(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        items = []
        async for item in engine:
            items.append(item)

        assert len(items) >= 1
        assert isinstance(items[0], dict)

    @pytest.mark.asyncio
    async def test_stream_processes_follow_up_requests(self):
        call_count = 0

        async def parse_with_follow(response) -> AsyncGenerator:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Request("https://example.com/page2", sid="default")
            yield {"page": call_count}

        spider = MockSpider()
        spider.parse = parse_with_follow  # type: ignore[assignment]
        engine = _make_engine(spider=spider)

        items = []
        async for item in engine:
            items.append(item)

        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_stream_items_not_stored_in_items_list(self):
        """When streaming, items go to the stream, not to engine._items."""
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        items = []
        async for item in engine:
            items.append(item)

        # Items were sent through stream, not appended to _items
        assert len(items) >= 1
        assert len(engine.items) == 0

    # ------------------------------------------------------------------
    # Streaming lifecycle / cleanup regression tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_full_consumption_cleans_up(self):
        """Full stream consumption should close session and call on_close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            session = TrackingSession()
            sm = SessionManager()
            sm.add("default", session)
            engine = CrawlerEngine(spider, sm, crawldir=tmpdir, interval=0)

            items = []
            async for item in engine._stream():
                items.append(item)

            assert len(items) >= 1
            assert session.closed is True
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_stream_break_without_aclose_cleans_up(self):
        """Breaking from async for without explicit aclose yields items.

        Note: Immediate session cleanup is not guaranteed after a bare
        async-for break because asyncio.Task.cancel() delivery requires an
        event-loop tick. Use explicit aclose() or contextlib.aclosing() for
        deterministic cleanup.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            session = TrackingSession()
            sm = SessionManager()
            sm.add("default", session)
            engine = CrawlerEngine(spider, sm, crawldir=tmpdir, interval=0)

            items = []
            async for item in engine._stream():
                items.append(item)
                if len(items) >= 1:
                    break

            assert len(items) == 1

    @pytest.mark.asyncio
    async def test_stream_break_with_explicit_aclose_cleans_up(self):
        """Breaking from async for and calling aclose() in the same task.

        async-for already calls aclose() on break, so the explicit aclose()
        is redundant but harmless. With the fix, await task in _stream()'s
        finally ensures cleanup completes before stream closure returns.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            session = TrackingSession()
            sm = SessionManager()
            sm.add("default", session)
            engine = CrawlerEngine(spider, sm, crawldir=tmpdir, interval=0)

            stream = engine._stream()
            items = []
            try:
                async for item in stream:
                    items.append(item)
                    if len(items) >= 1:
                        break
                await stream.aclose()
            except Exception:
                pass

            assert len(items) == 1
            assert session.closed is True
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_stream_aclosing_context_manager_cleans_up(self):
        """Using contextlib.aclosing() around stream consumption yields items.

        contextlib.aclosing() calls aclose() on context exit. With the fix,
        cleanup completes before stream closure returns.
        """
        import contextlib

        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            session = TrackingSession()
            sm = SessionManager()
            sm.add("default", session)
            engine = CrawlerEngine(spider, sm, crawldir=tmpdir, interval=0)

            items = []
            try:
                async with contextlib.aclosing(engine._stream()) as stream:
                    async for item in stream:
                        items.append(item)
                        if len(items) >= 1:
                            break
            except Exception:
                pass

            assert len(items) == 1
            assert session.closed is True
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_stream_consumer_cancellation_cleans_up(self):
        """Cancelling the consumer task should clean up crawl resources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            session = TrackingSession()
            sm = SessionManager()
            sm.add("default", session)
            engine = CrawlerEngine(spider, sm, crawldir=tmpdir, interval=0)

            stream = engine._stream()

            async def consumer():
                items = []
                async for item in stream:
                    items.append(item)
                    if len(items) >= 1:
                        break
                return items

            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(consumer)
                    await anyio.sleep(0.2)
                    tg.cancel_scope.cancel()
            except Exception:
                pass

            assert session.closed is True
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_stream_exception_in_consumer_cleans_up(self):
        """An exception raised inside the consumer loop should clean up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            session = TrackingSession()
            sm = SessionManager()
            sm.add("default", session)
            engine = CrawlerEngine(spider, sm, crawldir=tmpdir, interval=0)

            stream = engine._stream()
            items = []
            try:
                async for item in stream:
                    items.append(item)
                    if len(items) >= 1:
                        raise ValueError("consumer error")
            except ValueError:
                pass
            # Explicitly close the generator. With the fix, _stream()'s
            # finally awaits the producer task, so cleanup completes before
            # aclose() returns — no sleep needed.
            await stream.aclose()

            assert len(items) == 1
            assert session.closed is True
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_stream_active_tasks_return_to_zero(self):
        """After stream consumption (full or early exit), active tasks should be zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spider = MockSpider()
            session = TrackingSession()
            sm = SessionManager()
            sm.add("default", session)
            engine = CrawlerEngine(spider, sm, crawldir=tmpdir, interval=0)

            items = []
            async for item in engine._stream():
                items.append(item)
                if len(items) >= 1:
                    break

            assert engine._active_tasks == 0


# ---------------------------------------------------------------------------
# Tests: public Spider.stream() API
# ---------------------------------------------------------------------------


class TestSpiderStream:
    """Tests for the public Spider.stream() API.

    These tests verify that explicitly closing Spider.stream() (via break,
    aclose(), or contextlib.aclosing()) properly closes the inner
    CrawlerEngine._stream() generator, ensuring session cleanup and
    on_close() execution.
    """

    class StreamTestSpider(Spider):
        """Lightweight Spider subclass for stream() testing.

        Bypasses Spider.__init__() to avoid Playwright/browser dependencies
        while still exercising the real Spider.stream() method.
        """

        name = "stream_test_spider"

        def __init__(self, session, crawldir=None, block_event=None):
            self.name = "stream_test_spider"
            self.logger = logging.getLogger("test_stream_spider")
            self.logger.handlers.clear()
            self.logger.propagate = False
            self._session_manager = SessionManager()
            self._session_manager.add("default", session, default=True)
            self.crawldir = Path(crawldir) if crawldir else None
            self._interval = 0
            self._engine = None
            self.development_mode = False
            self.development_cache_dir = None
            self.concurrent_requests = 4
            self.concurrent_requests_per_domain = 0
            self.download_delay = 0.0
            self.max_blocked_retries = 3
            self.autothrottle_enabled = False
            self.autothrottle_start_delay = 5.0
            self.autothrottle_max_delay = 60.0
            self.autothrottle_target_concurrency = None
            self.autothrottle_block_backoff = True
            self.allowed_domains = set()
            self.fp_include_kwargs = False
            self.fp_include_headers = False
            self.fp_keep_fragments = False
            self.robots_txt_obey = False
            self.start_urls = ["https://example.com"]
            self.on_start_calls = []
            self.on_close_calls = 0
            self.scraped_items = []
            self.block_event = block_event or asyncio.Event()
            self.on_close_should_raise = False

            class _LogCounterStub:
                def get_counts(self):
                    return {"debug": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}

            self._log_counter = _LogCounterStub()

        def configure_sessions(self, session_manager: SessionManager):
            pass

        async def on_start(self, resuming: bool = False) -> None:
            self.on_start_calls.append({"resuming": resuming})

        async def on_close(self) -> None:
            self.on_close_calls += 1
            if self.on_close_should_raise:
                raise RuntimeError("on_close() raised")

        async def on_error(self, request: Request, error: Exception) -> None:
            pass

        async def on_scraped_item(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
            self.scraped_items.append(item)
            return item

        async def is_blocked(self, response) -> bool:
            return False

        async def retry_blocked_request(self, request: Request, response) -> Request:
            return request

        async def start_requests(self) -> AsyncGenerator[Request, None]:
            yield Request("https://example.com/page1", sid="default")

        async def parse(self, response) -> AsyncGenerator[Dict[str, Any] | None, None]:
            yield {"url": str(response)}

    @pytest.mark.asyncio
    async def test_spider_stream_break_closes_inner_generator(self):
        """Breaking from Spider.stream() closes the inner _stream() generator.

        Without the aclosing() fix in Spider.stream(), breaking from the
        outer generator does NOT close the inner CrawlerEngine._stream()
        generator, leaving sessions open and on_close() uncalled.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TrackingSession()
            spider = self.StreamTestSpider(session, crawldir=tmpdir)

            items = []
            stream = spider.stream()
            try:
                async for item in stream:
                    items.append(item)
                    if len(items) >= 1:
                        break
                await stream.aclose()
            except Exception:
                pass

            assert len(items) == 1
            assert session.closed is True, "Session should be closed after break"
            assert spider.on_close_calls == 1, "on_close() should be called once"
            assert spider._engine is None, "Engine reference should be cleared"

    @pytest.mark.asyncio
    async def test_spider_stream_on_close_raise_is_not_swallowed(self):
        """on_close() raising is not silently swallowed; session still closes.

        Verifies: on_close() called exactly once, session closed,
        exception visible (not silently lost).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TrackingSession()
            spider = self.StreamTestSpider(session, crawldir=tmpdir)
            spider.on_close_should_raise = True

            items = []
            stream = spider.stream()
            try:
                async for item in stream:
                    items.append(item)
                    if len(items) >= 1:
                        break
                await stream.aclose()
            except Exception:
                pass

            assert len(items) == 1
            assert spider.on_close_calls == 1, "on_close() should be called once"
            assert session.closed is True, "Session should be closed even when on_close() raises"

    @pytest.mark.asyncio
    async def test_spider_stream_full_consumption(self):
        """Full Spider.stream() consumption closes session and calls on_close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TrackingSession()
            spider = self.StreamTestSpider(session, crawldir=tmpdir)

            items = []
            async for item in spider.stream():
                items.append(item)

            assert len(items) >= 1
            assert session.closed is True
            assert spider.on_close_calls == 1
            assert spider._engine is None

    @pytest.mark.asyncio
    async def test_spider_stream_contextlib_aclosing(self):
        """contextlib.aclosing() around Spider.stream() closes everything."""
        import contextlib

        with tempfile.TemporaryDirectory() as tmpdir:
            session = TrackingSession()
            spider = self.StreamTestSpider(session, crawldir=tmpdir)

            items = []
            async with contextlib.aclosing(spider.stream()) as stream:
                async for item in stream:
                    items.append(item)
                    if len(items) >= 1:
                        break

            assert len(items) == 1
            assert session.closed is True
            assert spider.on_close_calls == 1

    # ---------------------------------------------------------------------------
    # Tests: producer failure and cleanup exception scenarios
    # ---------------------------------------------------------------------------

    class FailingStartRequestsSpider(StreamTestSpider):
        """Spider whose start_requests() raises before yielding any request."""

        name = "failing_start_requests_spider"

        async def start_requests(self):
            if False:
                yield  # pragma: no cover - makes this an async generator
            raise RuntimeError("controlled producer failure before first item")

    class FailingAfterItemSpider(StreamTestSpider):
        """Spider that yields one item, then fails on the next request."""

        name = "failing_after_item_spider"

        def __init__(self, session, crawldir=None):
            super().__init__(session, crawldir)
            self.call_count = 0

        async def start_requests(self):
            yield Request("https://example.com/page1", sid="default")
            yield Request("https://example.com/page2", sid="default")

        async def parse(self, response):
            self.call_count += 1
            if self.call_count == 1:
                yield {"url": str(response), "call": 1}
            else:
                raise RuntimeError("controlled producer failure after first item")

        async def on_error(self, request: Request, error: Exception) -> None:
            raise error

    class FailingCloseSession(TrackingSession):
        """Session whose __aexit__ raises an exception."""

        async def __aexit__(self, *args):
            self._is_alive = False
            self.closed = True
            raise RuntimeError("session close failed")

    class BlockingCloseSession(TrackingSession):
        """Session whose __aexit__ blocks on an Event.

        Mirrors real session behavior: _is_alive stays True during the blocked
        __aexit__ (set to False only after resource release completes), so an
        interrupted close() leaves the session appearing still-alive.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_event = asyncio.Event()
            self.aexit_started = asyncio.Event()
            self.aexit_completed = asyncio.Event()
            self.aexit_calls = 0

        async def __aexit__(self, *args):
            self.closed = True
            self.aexit_calls += 1
            self.aexit_started.set()
            await self.close_event.wait()
            self._is_alive = False
            self.aexit_completed.set()

    @pytest.mark.asyncio
    async def test_spider_stream_producer_failure_before_first_item(self):
        """A crawl-level RuntimeError from start_requests() reaches the consumer.

        Verifies that a genuine producer failure is not silently converted
        into normal end-of-stream.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TrackingSession()
            spider = self.FailingStartRequestsSpider(session, crawldir=tmpdir)

            stream = spider.stream()
            items = []
            consumer_exception = None
            try:
                async for item in stream:
                    items.append(item)
            except BaseException as e:
                consumer_exception = e

            await asyncio.sleep(0)

            assert len(items) == 0, "No items should be received before failure"
            assert consumer_exception is not None, "Consumer should receive producer exception"
            assert "controlled producer failure before first item" in str(consumer_exception)
            assert session.closed is True, "Session should be closed"
            assert spider.on_close_calls == 1, "on_close() should be called once"

    @pytest.mark.asyncio
    async def test_spider_stream_producer_failure_after_item(self):
        """Consumer receives one item, then the producer failure exception.

        Verifies: item delivered, subsequent failure, exception propagated,
        session cleanup completes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TrackingSession()
            spider = self.FailingAfterItemSpider(session, crawldir=tmpdir)

            stream = spider.stream()
            items = []
            consumer_exception = None
            try:
                async for item in stream:
                    items.append(item)
            except BaseException as e:
                consumer_exception = e

            await asyncio.sleep(0)

            assert len(items) == 1, "Consumer should receive exactly 1 item"
            assert consumer_exception is not None, "Consumer should receive producer exception"
            # Producer failure is wrapped in BaseExceptionGroup by anyio task group
            exc_group = consumer_exception
            found = any(
                "controlled producer failure after first item" in str(sub_exc)
                for sub_exc in getattr(exc_group, "exceptions", [exc_group])
            )
            assert found, f"Expected producer failure in exception group, got: {exc_group}"
            assert session.closed is True, "Session should be closed"
            assert spider.on_close_calls == 1, "on_close() should be called once"

    @pytest.mark.asyncio
    async def test_spider_stream_session_close_exception(self):
        """Session close failure is logged but does not swallow the crawl exception.

        Verifies: original crawl exception reaches consumer, session close
        attempted, on_close() called exactly once, no orphaned tasks.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.FailingCloseSession()
            spider = self.FailingStartRequestsSpider(session, crawldir=tmpdir)

            stream = spider.stream()
            items = []
            consumer_exception = None
            try:
                async for item in stream:
                    items.append(item)
            except BaseException as e:
                consumer_exception = e

            await asyncio.sleep(0)

            assert len(items) == 0
            assert consumer_exception is not None
            # Crawl exception may be wrapped or chained; walk __context__ to find it
            exc = consumer_exception
            found_crawl = False
            while exc is not None:
                if "controlled producer failure before first item" in str(exc):
                    found_crawl = True
                    break
                exc = getattr(exc, "__context__", None)
            assert found_crawl, f"Crawl exception lost in chain: {consumer_exception}"
            assert session.closed is True, "Session close should have been attempted"
            assert spider.on_close_calls == 1, "on_close() should be called once"

    @pytest.mark.asyncio
    async def test_spider_stream_blocked_cleanup_during_close(self):
        """Explicit stream closure waits for blocked session cleanup to complete.

        Verifies deterministic cleanup: aclose() does not return while
        session close() is blocked on an Event.
        """
        close_event = asyncio.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.BlockingCloseSession()
            session.close_event = close_event  # Wire the same Event object
            spider = self.StreamTestSpider(session, crawldir=tmpdir)

            stream = spider.stream()
            items = []
            aclose_completed = asyncio.Event()

            async def do_close():
                try:
                    async for item in stream:
                        items.append(item)
                        if len(items) >= 1:
                            break
                    await stream.aclose()
                finally:
                    aclose_completed.set()

            closer = asyncio.create_task(do_close())

            # Wait for item to be consumed
            while len(items) == 0:
                await asyncio.sleep(0.01)

            # Now closer is at aclose(), blocked in session close()
            assert session.closed is True, "Session cleanup should have started"
            assert not close_event.is_set(), "Close should be blocked on event"
            assert not aclose_completed.is_set(), "aclose() should still be pending"

            # Release the blocked cleanup
            close_event.set()
            await asyncio.wait_for(closer, timeout=5.0)

            assert aclose_completed.is_set(), "aclose() should complete after cleanup released"
            assert session.closed is True
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_spider_stream_second_cancellation_during_blocked_cleanup(self):
        """Second cancellation during blocked cleanup - shielded cleanup completes.

        Verifies that asyncio.shield() protects session_manager.close()
        from consumer cancellation, so the shielded operation can outlive
        the producer/consumer task.

        close_event is released right after cancelling so the engine's
        run() exception handler can complete its second SessionManager.close()
        without deadlocking on the blocked __aexit__.
        """
        close_event = asyncio.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.BlockingCloseSession()
            session.close_event = close_event  # Wire the same Event object
            spider = self.StreamTestSpider(session, crawldir=tmpdir)

            stream = spider.stream()
            items = []
            aclose_completed = asyncio.Event()

            async def do_close():
                try:
                    async for item in stream:
                        items.append(item)
                        if len(items) >= 1:
                            break
                    await stream.aclose()
                finally:
                    aclose_completed.set()

            closer = asyncio.create_task(do_close())

            # Wait for item
            while len(items) == 0:
                await asyncio.sleep(0.01)

            # Cancel the consumer while cleanup is blocked
            closer.cancel()

            # Release the blocked cleanup so the engine's run() exception
            # handler can complete its second SessionManager.close()
            close_event.set()

            try:
                await closer
            except asyncio.CancelledError:
                pass

            # Cleanup started and completed after release
            assert session.closed is True, "Session cleanup should have started"
            assert session._is_alive is False, "Cleanup completed - _is_alive now False"
            assert spider.on_close_calls == 1, "on_close() should be called once"

    @pytest.mark.asyncio
    async def test_spider_stream_consumer_cancellation_propagates_during_blocked_cleanup(self):
        """Consumer cancellation propagates when aclose() is blocked on cleanup.

        When the consumer task is cancelled while aclose() is blocked on
        producer cleanup, the CancelledError must reach the consumer's
        except handler — not be swallowed by _stream()'s finally block.

        Deterministic sequence:
        1. Start Spider.stream().
        2. Confirm the producer has started.
        3. Receive the first streamed item.
        4. Begin explicit aclose() in a consumer task.
        5. Confirm session cleanup has entered its blocked state.
        6. Cancel the consumer task.
        7. Record consumer and producer task states immediately after cancellation.
        8. Release close_event.
        9. Await completion with a bounded timeout.
        10. Verify whether cancellation propagated or was swallowed.

        Verifies:
        - Producer has started (consumer received at least 1 item)
        - Consumer performs explicit stream closure (aclose())
        - Cleanup is blocked using asyncio.Event
        - Consumer receives external cancellation
        - Cancellation is not silently converted into successful completion
        - Cleanup behavior is observable (session closed, on_close called once)
        - No task is unintentionally abandoned
        """
        close_event = asyncio.Event()

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.BlockingCloseSession()
            session.close_event = close_event  # Wire the same Event object
            spider = self.StreamTestSpider(session, crawldir=tmpdir)

            stream = spider.stream()
            items = []
            consumer_got_cancelled = False
            consumer_reached_aclose = asyncio.Event()

            async def consumer():
                nonlocal consumer_got_cancelled
                try:
                    async for item in stream:
                        items.append(item)
                        if len(items) >= 1:
                            break
                    consumer_reached_aclose.set()
                    await stream.aclose()
                except asyncio.CancelledError:
                    consumer_got_cancelled = True
                    raise

            consumer_task = asyncio.create_task(consumer())

            # Wait for item to be consumed (producer has started)
            while len(items) == 0:
                await asyncio.sleep(0.01)

            # Wait for consumer to reach aclose() (cleanup is blocked)
            await asyncio.wait_for(consumer_reached_aclose.wait(), timeout=5.0)

            # Verify cleanup started but is blocked
            assert session.closed is True, "Session cleanup should have started"
            assert not close_event.is_set(), "Close should be blocked on event"

            # Step 6: Cancel the consumer while cleanup is blocked
            consumer_task.cancel()

            # Step 7: Record task states immediately after cancellation
            consumer_task_cancelled_after_cancel = consumer_task.cancelled()
            consumer_task_done_after_cancel = consumer_task.done()

            # Step 8: Release close_event to unblock cleanup
            close_event.set()

            # Step 9: Await completion with a bounded timeout
            try:
                await asyncio.wait_for(consumer_task, timeout=5.0)
                consumer_task_await_result = "completed"
            except asyncio.CancelledError:
                consumer_task_await_result = "cancelled"
            except asyncio.TimeoutError:
                consumer_task_await_result = "timeout"

            # Give cleanup a moment to finish
            await asyncio.sleep(0.1)

        # Step 10: Verify whether cancellation propagated or was swallowed
        assert consumer_got_cancelled, (
            "Consumer's CancelledError handler must fire; "
            f"consumer_task state after cancel: cancelled={consumer_task_cancelled_after_cancel}, "
            f"done={consumer_task_done_after_cancel}, "
            f"await result={consumer_task_await_result}"
        )
        assert consumer_task.cancelled(), f"Consumer task must be cancelled; await result={consumer_task_await_result}"
        assert session.closed is True
        assert spider.on_close_calls == 1, "on_close() should be called exactly once"

    @pytest.mark.asyncio
    async def test_session_manager_second_close_after_interrupt(self):
        """Second SessionManager.close() after interrupted shutdown completes unfinished cleanup.

        When close() is interrupted during session.__aexit__(), the cancelled
        caller's CancelledError propagates but the shared cleanup task continues
        running (shielded from caller cancellation).  A subsequent close() awaits
        the same task instead of re-entering __aexit__().  Real sessions keep
        _is_alive=True during blocked __aexit__ (set to False only after resource
        release completes).

        We call SessionManager.close() directly (not through spider.stream())
        to avoid the engine's run() exception handler re-invoking close().

        Verifies:
        - Cleanup has NOT completed after interruption (_is_alive still True)
        - Cancelled caller's CancelledError propagates (not swallowed)
        - _started is True after cancellation (cleanup task still running)
        - Second close() awaits the same in-flight task (no duplicate __aexit__)
        - Session cleanup completes after close_event is released
        - __aexit__ called exactly once (the shared task owns it)
        - No cleanup task remains pending after completion
        """
        close_event = asyncio.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.BlockingCloseSession()
            session.close_event = close_event
            spider = self.StreamTestSpider(session, crawldir=tmpdir)
            sm = spider._session_manager

            # Simulate post-start state: session alive, manager started
            session._is_alive = True
            sm._started = True

            # Step 1: Start session shutdown - blocks inside __aexit__
            close_task = asyncio.create_task(sm.close())

            # Step 2: Wait for __aexit__ to enter blocked state
            await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

            # Confirm cleanup has NOT completed
            assert session.closed is True, "__aexit__ set closed=True"
            assert not close_event.is_set(), "Close blocked on event"
            assert session._is_alive is True, "Cleanup not completed - _is_alive still True"
            assert session.aexit_calls == 1, "__aexit__ called once so far"

            # Step 3: Cancel the first close caller
            close_task.cancel()
            try:
                await close_task
            except asyncio.CancelledError:
                pass

            # Cancelled caller's CancelledError propagates. _started is True
            # because the shared cleanup task is still running (hasn't reached
            # its finally block yet). _is_alive is still True because resource
            # release did not complete.
            assert sm._started is True, "_started True while cleanup task still running"
            assert session._is_alive is True, "Cleanup still not completed after cancel"
            assert session.aexit_calls == 1, "__aexit__ still called once after cancel"

            # Step 4: Second close() awaits the same in-flight task instead of
            # re-entering __aexit__().  The shared task owns __aexit__ execution.
            close_task2 = asyncio.create_task(sm.close())

            # __aexit__ still called exactly once - no duplicate entry by retry
            assert session.aexit_calls == 1, (
                f"Second close() must await the same task, not retry __aexit__; "
                f"expected 1 call, got {session.aexit_calls}"
            )

            # Step 5: Release the cleanup Event - the shared task completes
            close_event.set()
            await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)
            await close_task2

            # Cleanup completed
            assert session.closed is True
            assert session._is_alive is False, "Cleanup completed - _is_alive now False"

            # Both callers finished; no orphaned tasks
            assert close_task.done() and close_task.cancelled()
            assert close_task2.done() and not close_task2.cancelled()
            assert session.aexit_calls == 1, "__aexit__ called exactly once (shared task owns it)"
            assert sm._started is False, "_started False after cleanup completed"
            assert sm._close_done is True, "_close_done True after successful cleanup"

    @pytest.mark.asyncio
    async def test_session_manager_concurrent_close_calls(self):
        """Concurrent SessionManager.close() calls do not duplicate session shutdown.

        When close() is called concurrently from two tasks while the first close()
        is blocked inside __aexit__, the second close() awaits the same in-flight
        cleanup task instead of re-entering __aexit__(). Real sessions keep
        _is_alive=True during blocked __aexit__, so without coordination the
        second close() would enter __aexit__ concurrently.

        We call SessionManager.close() directly (not through spider.stream())
        to avoid the engine's run() exception handler re-invoking close().

        Verifies:
        - First close() blocks inside __aexit__ with _is_alive still True
        - Concurrent close() does NOT enter __aexit__ concurrently (deduplicated)
        - __aexit__ called exactly once total
        - Cleanup completes after close_event is released
        - Both close callers reach their expected terminal states
        - No cleanup task remains pending
        """
        close_event = asyncio.Event()
        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.BlockingCloseSession()
            session.close_event = close_event
            spider = self.StreamTestSpider(session, crawldir=tmpdir)
            sm = spider._session_manager

            # Simulate post-start state: session alive, manager started
            session._is_alive = True
            sm._started = True

            # Step 1: Start first close() - blocks inside __aexit__
            close1 = asyncio.create_task(sm.close())

            # Step 2: Wait for first __aexit__ to enter blocked state
            await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)

            # Confirm first close() is blocked with _is_alive still True
            assert session._is_alive is True, "First close() blocked - _is_alive still True"
            assert session.aexit_calls == 1, "__aexit__ called once by first close()"
            assert not close_event.is_set(), "Cleanup blocked on event"

            # Step 3: Start second close() concurrently
            close2 = asyncio.create_task(sm.close())

            # Give the second caller a moment to attach to the in-flight task
            await asyncio.sleep(0.1)

            # __aexit__ still called exactly once - no duplicate/concurrent entry
            assert session.aexit_calls == 1, (
                f"Concurrent close() must not re-enter __aexit__; "
                f"got {session.aexit_calls} calls"
            )

            # Step 4: Release cleanup - the shared task completes
            close_event.set()
            await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)

            # Wait for both close() callers to finish
            await asyncio.gather(close1, close2, return_exceptions=True)

            assert session.closed is True
            assert session._is_alive is False, "Cleanup completed - _is_alive now False"

            # Both callers finished successfully; no orphaned tasks
            assert close1.done() and not close1.cancelled()
            assert close2.done() and not close2.cancelled()
            assert session.aexit_calls == 1, "__aexit__ called exactly once total"


# ---------------------------------------------------------------------------
# Tests: Spider.stream() teardown inside ordinary asyncio tasks
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("producer_wait")
class TestSpiderStreamTeardown:
    """Closing or cancelling Spider.stream() from an ordinary asyncio task.

    CrawlerEngine._stream() cancels its producer task when the consumer stops
    early, then has to tell that cancellation apart from a cancellation of the
    consumer itself.  Python 3.11+ answers with Task.cancelling(); Python 3.10
    has no such method, so _stream() waits for the producer differently there.

    Every consumer here runs in a task made by asyncio.create_task().  On
    Python 3.10 pytest-asyncio drives the test function in a backported Task
    that has cancelling(), which hides what applications get on Python 3.10.
    Each test runs twice: on the path the interpreter selects, and with the
    Python 3.10 path forced, so that every interpreter covers that path.
    """

    class SlowSession(TrackingSession):
        """Keeps the crawl in flight while the consumer stops."""

        async def fetch(self, url: str, **kwargs):
            await asyncio.sleep(0.02)
            return await super().fetch(url, **kwargs)

    class LongCrawlSpider(TestSpiderStream.StreamTestSpider):
        """Has far more pages than any consumer here reads."""

        async def start_requests(self) -> AsyncGenerator[Request, None]:
            for page in range(200):
                yield Request(f"https://example.com/page{page}", sid="default")

    @pytest.fixture(params=["default", "py310"])
    def producer_wait(self, request, monkeypatch):
        """How _stream() waits for its producer: the interpreter's own path, or the Python 3.10 path."""
        if request.param == "py310":
            monkeypatch.setattr("scrapling.spiders.engine._task_cancelling", lambda task: None)
        return request.param

    @staticmethod
    def _unfinished_tasks() -> list:
        return [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("how", ["aclose", "aclosing"])
    async def test_early_close_of_running_stream_returns_cleanly(self, how):
        """Stopping a stream whose crawl is still running finishes cleanup, then returns normally."""
        import contextlib

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.SlowSession()
            spider = self.LongCrawlSpider(session, crawldir=tmpdir)
            items = []

            async def consumer():
                stream = spider.stream()
                if how == "aclose":
                    async for item in stream:
                        items.append(item)
                        break
                    await stream.aclose()
                else:
                    async with contextlib.aclosing(stream):
                        async for item in stream:
                            items.append(item)
                            break

            await asyncio.wait_for(asyncio.create_task(consumer()), timeout=5.0)

            assert len(items) == 1
            assert session.closed is True
            assert spider.on_close_calls == 1
            assert self._unfinished_tasks() == [], "no task may be left running"

    @pytest.mark.asyncio
    async def test_consumer_cancelled_while_iterating_ends_cancelled(self):
        """Cancelling a consumer that is waiting for the next item cancels it, after cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.SlowSession()
            spider = self.LongCrawlSpider(session, crawldir=tmpdir)
            got_item = asyncio.Event()

            async def consumer():
                async for _ in spider.stream():
                    got_item.set()

            task = asyncio.create_task(consumer())
            await asyncio.wait_for(got_item.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5.0)

            assert task.cancelled()
            assert session.closed is True
            assert spider.on_close_calls == 1
            assert self._unfinished_tasks() == [], "no task may be left running"

    @pytest.mark.asyncio
    async def test_close_waits_for_blocked_cleanup(self):
        """aclose() returns only once blocked session cleanup is released, and then returns normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TestSpiderStream.BlockingCloseSession()
            spider = TestSpiderStream.StreamTestSpider(session, crawldir=tmpdir)

            async def consumer():
                stream = spider.stream()
                async for _ in stream:
                    break
                await stream.aclose()

            task = asyncio.create_task(consumer())
            try:
                await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)
                await asyncio.sleep(0.05)
                assert not task.done(), "aclose() must keep waiting while cleanup is blocked"
            finally:
                session.close_event.set()  # never leave cleanup blocked, even if this test fails
            await asyncio.wait_for(task, timeout=5.0)

            assert session._is_alive is False
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_consumer_cancelled_during_blocked_close_ends_cancelled(self):
        """A consumer cancelled while aclose() waits on cleanup is cancelled; it must not finish normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TestSpiderStream.BlockingCloseSession()
            spider = TestSpiderStream.StreamTestSpider(session, crawldir=tmpdir)

            async def consumer():
                stream = spider.stream()
                async for _ in stream:
                    break
                await stream.aclose()

            task = asyncio.create_task(consumer())
            try:
                await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)
                task.cancel()
            finally:
                session.close_event.set()  # never leave cleanup blocked, even if this test fails
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5.0)

            assert task.cancelled()
            await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)
            assert session._is_alive is False
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_producer_failure_during_early_close_reaches_consumer(self):
        """An error the producer raises while it shuts down is raised to the consumer, never dropped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = self.SlowSession()
            spider = self.LongCrawlSpider(session, crawldir=tmpdir)
            spider.on_close_should_raise = True

            async def consumer():
                stream = spider.stream()
                async for _ in stream:
                    break
                await stream.aclose()

            task = asyncio.create_task(consumer())
            with pytest.raises(RuntimeError, match=r"on_close\(\) raised"):
                await asyncio.wait_for(task, timeout=5.0)

            assert session.closed is True
            assert spider.on_close_calls == 1

    @pytest.mark.asyncio
    async def test_cancel_scope_releases_consumer_while_cleanup_is_blocked(self):
        """A cancelled AnyIO scope ends the consumer although session cleanup has not finished.

        AnyIO repeats a scope's cancellation until the task leaves the scope, and
        recognises its own CancelledError by the message on it.  So the consumer
        must pass every request on to the producer, and must re-raise the error
        it was given; otherwise a timeout around the stream could never fire.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TestSpiderStream.BlockingCloseSession()
            spider = TestSpiderStream.StreamTestSpider(session, crawldir=tmpdir)
            scope = anyio.CancelScope()

            async def consumer():
                with scope:
                    stream = spider.stream()
                    async for _ in stream:
                        break
                    await stream.aclose()

            task = asyncio.create_task(consumer())
            try:
                await asyncio.wait_for(session.aexit_started.wait(), timeout=5.0)
                scope.cancel()
                finished, _ = await asyncio.wait({task}, timeout=5.0)

                assert finished, "the consumer must leave the cancelled scope while cleanup is still blocked"
                assert session._is_alive is True, "cleanup has not finished yet"
            finally:
                session.close_event.set()  # never leave cleanup blocked, even if this test fails
                await asyncio.wait({task}, timeout=5.0)

            assert not task.cancelled() and task.exception() is None, "the scope absorbs its own cancellation"
            assert scope.cancelled_caught is True
            await asyncio.wait_for(session.aexit_completed.wait(), timeout=5.0)
            assert session._is_alive is False, "the shared cleanup still completes"
            assert spider.on_close_calls == 1


# ---------------------------------------------------------------------------
# Tests: pause during crawl
# ---------------------------------------------------------------------------


class TestPauseDuringCrawl:
    @pytest.mark.asyncio
    async def test_pause_stops_crawl_gracefully(self):
        processed = 0

        async def slow_parse(response) -> AsyncGenerator:
            nonlocal processed
            processed += 1
            # Yield more requests to keep the crawl going
            if processed <= 2:
                yield Request(f"https://example.com/p{processed + 1}", sid="default")
            yield {"n": processed}

        spider = MockSpider()
        spider.parse = slow_parse  # type: ignore[assignment]
        engine = _make_engine(spider=spider)

        # Request pause immediately - the engine will stop as soon as active tasks complete
        engine._pause_requested = True

        stats = await engine.crawl()
        # Should stop without processing everything
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_pause_with_checkpoint_sets_paused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parse_count = 0

            async def parse_and_pause(response) -> AsyncGenerator:
                nonlocal parse_count
                parse_count += 1
                # Request pause after first request, but yield follow-ups
                if parse_count == 1:
                    engine.request_pause()
                    yield Request("https://example.com/p2", sid="default")
                yield {"n": parse_count}

            spider = MockSpider()
            spider.parse = parse_and_pause  # type: ignore[assignment]
            engine = _make_engine(spider=spider, crawldir=tmpdir)

            await engine.crawl()

            assert engine.paused is True

    @pytest.mark.asyncio
    async def test_pause_without_checkpoint_does_not_set_paused(self):
        spider = MockSpider()
        engine = _make_engine(spider=spider)

        engine._pause_requested = True

        await engine.crawl()

        assert engine.paused is False


# ---------------------------------------------------------------------------
# Tests: _prefetch_robots_txt
# ---------------------------------------------------------------------------


class TestPrefetchRobotsTxt:
    """_prefetch_robots_txt warms the robots.txt cache before the crawl loop."""

    @staticmethod
    def _make_counting_fetch():
        """Return (fetch_fn, calls_list) where calls_list records every (url, sid) pair."""
        calls: list[tuple[str, str]] = []

        async def _fetch(url: str, sid: str):
            calls.append((url, sid))
            return MockResponse(status=200, body=b"", url=url)

        return _fetch, calls

    @pytest.mark.asyncio
    async def test_prefetch_uses_start_urls(self):
        fetch_fn, calls = self._make_counting_fetch()
        spider = MockSpider(robots_txt_obey=True, start_urls=["https://example.com/page1"])
        engine = _make_engine(spider=spider)
        engine._robots_manager = RobotsTxtManager(fetch_fn)

        await engine._prefetch_robots_txt()

        assert len(calls) == 1
        assert calls[0][0] == "https://example.com/robots.txt"

    @pytest.mark.asyncio
    async def test_prefetch_noop_when_robots_disabled(self):
        fetch_fn, calls = self._make_counting_fetch()
        spider = MockSpider(robots_txt_obey=False)
        engine = _make_engine(spider=spider)

        assert engine._robots_manager is None

        await engine._prefetch_robots_txt()

        assert calls == []

    @pytest.mark.asyncio
    async def test_prefetch_noop_when_start_urls_empty(self):
        fetch_fn, calls = self._make_counting_fetch()
        spider = MockSpider(robots_txt_obey=True, start_urls=[])
        engine = _make_engine(spider=spider)
        engine._robots_manager = RobotsTxtManager(fetch_fn)

        await engine._prefetch_robots_txt()

        assert calls == []

    @pytest.mark.asyncio
    async def test_prefetch_deduplicates_same_domain_in_start_urls(self):
        fetch_fn, calls = self._make_counting_fetch()
        spider = MockSpider(robots_txt_obey=True, start_urls=["https://example.com/a", "https://example.com/b"])
        engine = _make_engine(spider=spider)
        engine._robots_manager = RobotsTxtManager(fetch_fn)

        await engine._prefetch_robots_txt()

        # set of Request.domain values deduplicates to one task per domain
        assert len(calls) == 1
        assert calls[0][0] == "https://example.com/robots.txt"
