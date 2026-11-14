from unittest.mock import AsyncMock, Mock

import pytest

from scrapling.parser import Selector
from scrapling.engines._browsers._base import AsyncSession, SyncSession
from scrapling.engines.toolbelt.convertor import ResponseFactory, Response


HTML_WITH_ACCENTED_TEXT = "<html><body>Bürger</body></html>"
LATIN9_CONTENT_TYPE = "text/html; charset=iso-8859-15"


def make_sync_playwright_response():
    response = Mock()
    response.url = "https://example.com"
    response.status = 200
    response.status_text = "OK"
    response.headers = {"content-type": LATIN9_CONTENT_TYPE}
    response.all_headers = Mock(return_value={"content-type": LATIN9_CONTENT_TYPE})
    response.body = Mock(return_value=HTML_WITH_ACCENTED_TEXT.encode("iso-8859-15"))
    response.request.method = "GET"
    response.request.headers = {}
    response.request.all_headers = Mock(return_value={})
    response.request.redirected_from = None
    return response


def make_sync_page():
    page = Mock()
    page.url = "https://example.com"
    page.content = Mock(return_value=HTML_WITH_ACCENTED_TEXT)
    page.context.cookies = Mock(return_value=[])
    return page


def make_async_playwright_response():
    response = Mock()
    response.url = "https://example.com"
    response.status = 200
    response.status_text = "OK"
    response.headers = {"content-type": LATIN9_CONTENT_TYPE}
    response.all_headers = AsyncMock(return_value={"content-type": LATIN9_CONTENT_TYPE})
    response.body = AsyncMock(return_value=HTML_WITH_ACCENTED_TEXT.encode("iso-8859-15"))
    response.request.method = "GET"
    response.request.headers = {}
    response.request.all_headers = AsyncMock(return_value={})
    response.request.redirected_from = None
    return response


def make_async_page():
    page = Mock()
    page.url = "https://example.com"
    page.content = AsyncMock(return_value=HTML_WITH_ACCENTED_TEXT)
    page.context.cookies = AsyncMock(return_value=[])
    return page


class TestResponseFactory:
    """Test ResponseFactory functionality"""

    @pytest.mark.parametrize("arguments,expected", [({}, "GET"), ({"method": "POST"}, "POST")])
    def test_response_keeps_request_method(self, arguments, expected):
        response = Response(
            url="https://example.com",
            content=b"saved",
            status=200,
            reason="OK",
            cookies={},
            headers={},
            request_headers={},
            **arguments,
        )

        assert response.method == expected
        assert not hasattr(response, "captured_xhr")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_response_handler_keeps_only_main_frame_navigation(self, asynchronous):
        main_frame = Mock()
        page_info = Mock(page=Mock(main_frame=main_frame))
        responses = [None]
        latest = None
        handler = (
            AsyncSession._create_response_handler(page_info, responses)
            if asynchronous
            else SyncSession._create_response_handler(page_info, responses)
        )
        for resource_type, navigation, main, captured in (
            ("document", True, True, True),
            ("xhr", False, True, False),
            ("fetch", False, True, False),
            ("image", False, True, False),
            ("document", False, True, False),
            ("document", True, False, False),
            ("document", True, True, True),
        ):
            response = Mock()
            response.request.resource_type = resource_type
            response.request.is_navigation_request.return_value = navigation
            response.request.frame = main_frame if main else Mock()
            result = handler(response)
            if result is not None:
                await result
            if captured:
                latest = response
            assert responses[0] is latest

    def test_response_from_curl(self):
        """Test creating response from curl_cffi response"""
        # Mock curl response
        mock_curl_response = Mock()
        mock_curl_response.url = "https://example.com"
        mock_curl_response.content = b"<html><body>Test</body></html>"
        mock_curl_response.status_code = 200
        mock_curl_response.reason = "OK"
        mock_curl_response.encoding = "utf-8"
        mock_curl_response.cookies = {"session": "abc"}
        mock_curl_response.headers = {"Content-Type": "text/html"}
        mock_curl_response.request.headers = {"User-Agent": "Test"}
        mock_curl_response.request.method = "GET"
        mock_curl_response.history = []

        response = ResponseFactory.from_http_request(mock_curl_response, {"adaptive": False})

        assert response.status == 200
        assert response.url == "https://example.com"
        assert isinstance(response, Response)

    def test_playwright_page_content_uses_utf8_encoding(self):
        """page.content() returns Unicode, so its encoded bytes are UTF-8."""
        mock_response = make_sync_playwright_response()
        mock_page = make_sync_page()

        response = ResponseFactory.from_playwright_response(
            mock_page,
            mock_response,
            mock_response,
            {"adaptive": False},
            collect_history=False,
        )

        assert response.encoding == "utf-8"
        assert "Bürger" in response.html_content
        assert "BÃ" not in response.html_content

    @pytest.mark.parametrize(
        "content_type",
        [LATIN9_CONTENT_TYPE, 'text/html; CHARSET = "iso-8859-15"', "text/html; charset = 'iso-8859-15'"],
    )
    def test_playwright_raw_response_keeps_header_encoding(self, content_type):
        """Raw response bytes should still use the charset from Content-Type."""
        mock_response = make_sync_playwright_response()
        mock_response.headers = {"content-type": content_type}
        mock_response.all_headers.return_value = mock_response.headers

        response = ResponseFactory.from_playwright_response(
            None,
            mock_response,
            mock_response,
            {"adaptive": False},
            collect_history=False,
        )

        assert response.encoding == "iso-8859-15"
        assert "Bürger" in response.html_content

    @pytest.mark.asyncio
    async def test_async_playwright_page_content_uses_utf8_encoding(self):
        """Async page.content() returns Unicode, so its encoded bytes are UTF-8."""
        mock_response = make_async_playwright_response()
        mock_page = make_async_page()

        response = await ResponseFactory.from_async_playwright_response(
            mock_page,
            mock_response,
            mock_response,
            {"adaptive": False},
            collect_history=False,
        )

        assert response.encoding == "utf-8"
        assert "Bürger" in response.html_content
        assert "BÃ" not in response.html_content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content_type",
        [LATIN9_CONTENT_TYPE, 'text/html; CHARSET = "iso-8859-15"', "text/html; charset = 'iso-8859-15'"],
    )
    async def test_async_playwright_raw_response_keeps_header_encoding(self, content_type):
        """Raw response bytes should still use the charset from Content-Type."""
        mock_response = make_async_playwright_response()
        mock_response.headers = {"content-type": content_type}
        mock_response.all_headers.return_value = mock_response.headers

        response = await ResponseFactory.from_async_playwright_response(
            None,
            mock_response,
            mock_response,
            {"adaptive": False},
            collect_history=False,
        )

        assert response.encoding == "iso-8859-15"
        assert "Bürger" in response.html_content

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_response_history_processing(self, asynchronous):
        make = make_async_playwright_response if asynchronous else make_sync_playwright_response
        make_method = AsyncMock if asynchronous else Mock
        first, middle, final = make(), make(), make()
        for native, path, status in ((first, "start", 301), (middle, "middle", 302), (final, "final", 200)):
            native.url = native.request.url = f"https://example.com/{path}"
            native.status = status
            native.status_text = ""
            native.request.response = make_method(return_value=native)
            native.request.all_headers.return_value = {"x-redirect": path}
        final.request.redirected_from = middle.request
        middle.request.redirected_from = first.request
        if asynchronous:
            response = await ResponseFactory.from_async_playwright_response(None, final, None, {})
        else:
            response = ResponseFactory.from_playwright_response(None, final, None, {})
        assert response.status == 200 and response.url == "https://example.com/final"
        assert [(item.url, item.status, item.reason) for item in response.history] == [
            ("https://example.com/start", 301, "Moved Permanently"),
            ("https://example.com/middle", 302, "Found"),
        ]
        assert [item.request_headers for item in response.history] == [
            {"x-redirect": "start"},
            {"x-redirect": "middle"},
        ]
        assert all(item.body == b"" and item.encoding == "iso-8859-15" for item in response.history)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
class TestRecordedResponseFactory:
    @staticmethod
    async def convert(native, asynchronous, **kwargs):
        if asynchronous:
            return await ResponseFactory.from_async_playwright_response(
                None, native, None, {}, collect_history=False, **kwargs
            )
        return ResponseFactory.from_playwright_response(None, native, None, {}, collect_history=False, **kwargs)

    @staticmethod
    def native(asynchronous, body=b"data", headers=None):
        native = make_async_playwright_response() if asynchronous else make_sync_playwright_response()
        native.headers = {"content-type": "text/plain"} if headers is None else headers
        native.all_headers.return_value = native.headers
        native.body.return_value = body
        return native

    async def test_saved_response_keeps_bytes_and_full_headers(self, asynchronous):
        native = self.native(asynchronous, b'\x00{"data": "saved"}\x00')
        native.all_headers.return_value = {**native.headers, "set-cookie": "session=value"}
        native.request.all_headers.return_value = {"cookie": "session=value"}
        metadata = {"proxy": "http://proxy"}

        response = await self.convert(native, asynchronous, max_body_bytes=100, meta=metadata)

        assert isinstance(response, Response)
        assert response.body == b'\x00{"data": "saved"}\x00'
        assert response.headers["set-cookie"] == "session=value"
        assert response.request_headers["cookie"] == "session=value"
        assert response.meta == {"proxy": "http://proxy"}
        assert metadata == {"proxy": "http://proxy"}

    @pytest.mark.parametrize("method,status", [("HEAD", 200), ("GET", 204), ("GET", 205), ("GET", 304)])
    @pytest.mark.parametrize("declared_length", [None, "1000000"])
    async def test_absent_bodies_are_not_read(self, asynchronous, method, status, declared_length):
        native = self.native(asynchronous, headers={})
        native.request.method, native.status = method, status
        if declared_length is not None:
            native.headers.update({"content-type": "application/octet-stream", "content-length": declared_length})

        response = await self.convert(native, asynchronous, max_body_bytes=100)

        assert response.body == b""
        assert response.meta == {}
        native.body.assert_not_called()

    async def test_content_length_skips_large_body(self, asynchronous):
        native = self.native(asynchronous, headers={"content-type": "text/plain", "content-length": "101"})

        response = await self.convert(native, asynchronous, max_body_bytes=100)

        assert response.body == b""
        assert response.meta == {"body_note": "too large; not saved."}
        native.body.assert_not_called()

    @pytest.mark.parametrize("content_type", ["application/octet-stream", "image/png", "application/x-custom", None])
    async def test_non_text_bodies_are_not_read_but_metadata_is_saved(self, asynchronous, content_type):
        headers = {} if content_type is None else {"content-type": content_type}
        native = self.native(asynchronous, b"\x00\xffbinary", headers)
        native.status, native.status_text, native.request.method = 206, "Partial Content", "POST"
        native.all_headers.return_value = {**headers, "x-saved": "response"}
        native.request.all_headers.return_value = {"x-saved": "request"}
        native.body.side_effect = AssertionError("Non-text body must not be read")

        response = await self.convert(native, asynchronous, max_body_bytes=100)

        assert response.body == b""
        assert response.meta == {"body_note": "Non-text body; not saved."}
        assert response.url == native.url and response.status == 206 and response.reason == "Partial Content"
        assert response.headers == {**headers, "x-saved": "response"}
        assert response.request_headers == {"x-saved": "request"}
        native.body.assert_not_called()

    @pytest.mark.parametrize(
        "content_type,body",
        [
            (' Text/Plain ; charset="utf-8"', "café".encode()),
            ("application/json", b'{"saved": true}'),
            ("application/xml", b"<saved>true</saved>"),
            ("application/javascript", b"const saved = true;"),
            ("application/x-javascript", b"var saved = true;"),
            ("application/graphql", b"{ saved }"),
            ("application/x-www-form-urlencoded", b"saved=true"),
            ("application/problem+json", b'{"saved": true}'),
            ("image/svg+xml", b"<svg></svg>"),
        ],
    )
    async def test_text_content_types_keep_the_complete_body(self, asynchronous, content_type, body):
        native = self.native(asynchronous, body, {"content-type": content_type})

        response = await self.convert(native, asynchronous, max_body_bytes=len(body))

        assert response.body == body
        assert response.headers == {"content-type": content_type}
        assert response.meta == {}
        native.body.assert_called_once_with()

    @pytest.mark.parametrize("declared_length", [None, "1", "invalid"])
    async def test_actual_size_rejects_whole_body(self, asynchronous, declared_length):
        native = self.native(asynchronous, b"12345")
        if declared_length is not None:
            native.headers["content-length"] = declared_length

        response = await self.convert(native, asynchronous, max_body_bytes=4)

        assert response.body == b""
        assert response.meta == {"body_note": "too large; not saved."}
        native.body.assert_called_once()

    @pytest.mark.parametrize("body,limit", [(b"1234", 4), (b"", 0)])
    @pytest.mark.parametrize("declared", [False, True])
    async def test_available_body_at_limit(self, asynchronous, body, limit, declared):
        native = self.native(asynchronous, body)
        if declared:
            native.headers["content-length"] = str(len(body))

        response = await self.convert(native, asynchronous, max_body_bytes=limit)

        assert response.body == body
        assert "body_note" not in response.meta

    @pytest.mark.parametrize("message", ["Response body is unavailable", ""])
    async def test_read_failure_is_not_an_empty_available_body(self, asynchronous, monkeypatch, message):
        native = self.native(asynchronous)
        native.body.side_effect = RuntimeError(message)
        logger = Mock()
        monkeypatch.setattr("scrapling.engines.toolbelt.convertor.log", logger)

        response = await self.convert(native, asynchronous, max_body_bytes=100)

        assert response.body == b""
        assert response.meta == {"body_note": message or "could not be saved."}
        logger.error.assert_not_called()

    async def test_header_failures_keep_partial_headers(self, asynchronous):
        native = self.native(asynchronous)
        native.all_headers.side_effect = RuntimeError("closed")
        native.request.all_headers.side_effect = RuntimeError("closed")
        native.request.headers = {"accept": "text/plain"}

        response = await self.convert(native, asynchronous, max_body_bytes=100)

        assert response.body == b"data"
        assert response.headers == {"content-type": "text/plain"}
        assert response.request_headers == {"accept": "text/plain"}
        assert response.meta == {"headers_partial": True, "request_headers_partial": True}

    @pytest.mark.parametrize(
        "body,content_type",
        [(b"\xff\x00\xfe", "text/plain"), (b"\xff\x00\xfe", "text/plain; charset=bogus")],
    )
    async def test_parser_errors_do_not_drop_saved_bytes(self, asynchronous, body, content_type):
        native = self.native(asynchronous, body, {"content-type": content_type})

        response = await self.convert(native, asynchronous, max_body_bytes=100)

        assert response.body == body
        assert "body_note" not in response.meta
        assert response.headers["content-type"] == content_type
        assert isinstance(response.html_content, str)
        if content_type.endswith("charset=bogus"):
            assert response.encoding == "utf-8" and response.get_all_text() == ""

    @pytest.mark.parametrize(
        "body,expected,encoding",
        [
            ("café €".encode(), "café €", "utf-8"),
            (b"caf\xe9 \xa4", "café €", "iso-8859-15"),
            (b"\xc3\xa9", "é", "utf-8"),
        ],
    )
    @pytest.mark.parametrize("recorded", [False, True])
    async def test_native_bytes_and_selected_text_encoding(self, asynchronous, body, expected, encoding, recorded):
        content_type = "text/plain; charset=iso-8859-15"
        native = self.native(asynchronous, body, {"content-type": content_type})

        response = await self.convert(native, asynchronous, max_body_bytes=100 if recorded else None)

        assert response.body == body
        assert response.headers["content-type"] == content_type
        assert response.encoding == (encoding if recorded else "iso-8859-15")
        assert response.get_all_text().strip() == (expected if recorded else body.decode("iso-8859-15"))
        assert response.meta == {}
        native.body.assert_called_once_with()

    async def test_recorded_utf8_text_ignores_unknown_declared_charset(self, asynchronous):
        body = '{"saved": "café €"}'.encode()
        content_type = "application/json; charset=bogus"
        native = self.native(asynchronous, body, {"content-type": content_type})

        response = await self.convert(native, asynchronous, max_body_bytes=100)

        assert response.body == body and response.headers["content-type"] == content_type
        assert response.encoding == "utf-8"
        assert response.get_all_text().strip() == '{"saved": "café €"}'
        assert response.json() == {"saved": "café €"}
        assert response.meta == {}

    async def test_unlimited_mode_does_not_add_recording_metadata(self, asynchronous):
        native = self.native(asynchronous, b"data", {"content-length": "10000000"})

        response = await self.convert(native, asynchronous)

        assert response.body == b"data"
        assert response.meta == {}
        native.body.assert_called_once()

    async def test_unlimited_mode_keeps_binary_bodies(self, asynchronous):
        body = b"\x89PNG\r\n\x1a\n\x00binary"
        native = self.native(asynchronous, body, {"content-type": "image/png"})

        response = await self.convert(native, asynchronous)

        assert response.body == body
        assert response.headers == {"content-type": "image/png"}
        assert response.meta == {}
        native.body.assert_called_once_with()

    @pytest.mark.parametrize("source", ["response", "request"])
    async def test_unlimited_mode_keeps_header_errors(self, asynchronous, source):
        native = self.native(asynchronous)
        target = native if source == "response" else native.request
        target.all_headers.side_effect = RuntimeError("closed")

        with pytest.raises(RuntimeError, match="closed"):
            await self.convert(native, asynchronous)

    async def test_unlimited_mode_keeps_body_error_behavior(self, asynchronous, monkeypatch):
        native = self.native(asynchronous)
        native.body.side_effect = RuntimeError("closed")
        logger = Mock()
        monkeypatch.setattr("scrapling.engines.toolbelt.convertor.log", logger)

        response = await self.convert(native, asynchronous)

        assert response.body == b""
        assert response.meta == {}
        logger.error.assert_called_once()

    async def test_unlimited_mode_keeps_encoding_errors(self, asynchronous):
        native = self.native(asynchronous, headers={"content-type": "text/plain; charset=bogus"})

        with pytest.raises(LookupError):
            await self.convert(native, asynchronous)

    @pytest.mark.parametrize("source", ["response", "request", "both"])
    async def test_body_error_stays_unavailable_when_headers_also_fail(self, asynchronous, source):
        native = self.native(asynchronous)
        native.body.side_effect = RuntimeError("Body is gone")
        if source in {"response", "both"}:
            native.all_headers.side_effect = RuntimeError("Response headers are gone")
        if source in {"request", "both"}:
            native.request.all_headers.side_effect = RuntimeError("Request headers are gone")
        response = await self.convert(native, asynchronous, max_body_bytes=100)
        assert response.body == b""
        assert response.meta["body_note"] == "Body is gone"
        assert response.meta.get("headers_partial", False) == (source in {"response", "both"})
        assert response.meta.get("request_headers_partial", False) == (source in {"request", "both"})
        assert response.headers == native.headers and response.request_headers == native.request.headers

    async def test_missing_response_raises_instead_of_returning_empty_data(self, asynchronous):
        with pytest.raises(ValueError, match="Failed to get a response"):
            await self.convert(None, asynchronous, max_body_bytes=100)

    async def test_factory_rejects_removed_xhr_capture_argument(self, asynchronous):
        native = self.native(asynchronous)
        with pytest.raises(TypeError, match="unexpected keyword argument 'xhr_captured'"):
            await self.convert(native, asynchronous, xhr_captured=[native])
        native.body.assert_not_called()


class TestErrorScenarios:
    """Test various error scenarios"""

    def test_invalid_html_handling(self):
        """Test handling of malformed HTML"""
        malformed_html = """
        <html>
            <body>
                <div>Unclosed div
                <p>Paragraph without closing tag
                <span>Nested unclosed
            </body>
        """

        # Should handle gracefully
        page = Selector(malformed_html)
        assert page is not None

        # Should still be able to select elements
        divs = page.css("div")
        assert len(divs) > 0

    def test_empty_responses(self):
        """Test handling of empty responses"""
        # Empty HTML
        page = Selector("")
        assert page is not None

        # Whitespace only
        page = Selector("   \n\t   ")
        assert page is not None

        # Null bytes
        page = Selector("Hello\x00World")
        assert "Hello" in page.get_all_text()


class TestNextPageUrl:
    """Deterministic tests for Response.next_page_url()."""

    @staticmethod
    def _make_response(html: str, url: str = "https://example.com/page") -> Response:
        return Response(
            url=url,
            content=html,
            status=200,
            reason="OK",
            cookies={},
            headers={},
            request_headers={},
        )

    # --- Priority 1: rel="next" ---

    def test_head_link_rel_next(self):
        html = '<html><head><link rel="next" href="/page/2"></head><body></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_anchor_rel_next(self):
        html = '<html><body><a rel="next" href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_area_rel_next(self):
        html = '<html><body><map><area rel="next" href="/page/2"></map></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_rel_next_case_insensitive(self):
        html = '<html><body><a rel="NEXT" href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_rel_next_multi_token(self):
        html = '<html><body><a rel="next archive" href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_rel_next_multi_token_reversed(self):
        html = '<html><body><a rel="archive next" href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_rel_not_next_is_ignored(self):
        html = '<html><body><a rel="prev" href="/page/0">Prev</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    # --- Relative URL resolution ---

    def test_relative_url_resolved(self):
        html = '<html><body><a rel="next" href="page/2">Next</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/sub/").next_page_url()
            == "https://example.com/sub/page/2"
        )

    def test_relative_url_with_dots(self):
        html = '<html><body><a rel="next" href="../page/2">Next</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/sub/deep/").next_page_url()
            == "https://example.com/sub/page/2"
        )

    # --- Priority 2: accessibility labels ---

    def test_aria_label_next(self):
        html = '<html><body><a aria-label="Next page" href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_aria_label_next_case_insensitive(self):
        html = '<html><body><a aria-label="NEXT" href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_aria_label_with_extra_text(self):
        html = '<html><body><a aria-label="Go to next page" href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_aria_label_no_href_is_skipped(self):
        html = '<html><body><a aria-label="Next">No href</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    # --- Priority 3: link text ---

    def test_link_text_next(self):
        html = '<html><body><a href="/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_link_text_gt(self):
        html = '<html><body><a href="/page/2">&gt;</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_link_text_double_gt(self):
        html = '<html><body><a href="/page/2">&raquo;</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_link_text_older(self):
        html = '<html><body><a href="/page/2">Older posts</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_link_text_next_page(self):
        html = '<html><body><a href="/page/2">Next page</a></body></html>'
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_generic_more_link_is_not_pagination(self):
        html = '<html><body><a href="/more">More</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    # --- Priority 4: numbered pagination ---

    def test_numbered_pagination_query_page(self):
        html = '<html><body><a href="?page=2">Page 2</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/?page=1").next_page_url()
            == "https://example.com/?page=2"
        )

    def test_numbered_pagination_path_page(self):
        html = '<html><body><a href="/page/2/">Page 2</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/page/1/").next_page_url()
            == "https://example.com/page/2/"
        )

    def test_numbered_pagination_lower_page_is_ignored(self):
        html = '<html><body><a href="?page=0">Page 0</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/?page=1").next_page_url()
            is None
        )

    def test_numbered_pagination_same_page_is_ignored(self):
        html = '<html><body><a href="?page=1">Page 1</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/?page=1").next_page_url()
            is None
        )

    def test_numbered_pagination_no_current_page(self):
        html = '<html><body><a href="?page=2">Page 2</a></body></html>'
        assert self._make_response(html, url="https://example.com/").next_page_url() is None

    # --- Precedence ---

    def test_rel_next_wins_over_text(self):
        html = (
            '<html><body>'
            '<a rel="next" href="/page/2">Next</a>'
            '<a href="/page/3">Next</a>'
            "</body></html>"
        )
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_aria_label_wins_over_text(self):
        html = (
            '<html><body>'
            '<a aria-label="Next" href="/page/2">Click</a>'
            '<a href="/page/3">Next</a>'
            "</body></html>"
        )
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    # --- Duplicates ---

    def test_duplicate_rel_next_candidates(self):
        html = (
            '<html><body>'
            '<a rel="next" href="/page/2">Next</a>'
            '<a rel="next" href="/page/2">Next</a>'
            "</body></html>"
        )
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    # --- URL validation ---

    def test_self_link_is_rejected(self):
        html = '<html><body><a rel="next" href="?">Next</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/?").next_page_url()
            is None
        )

    def test_self_link_with_trailing_slash_is_rejected(self):
        html = '<html><body><a rel="next" href="/page/">Next</a></body></html>'
        assert (
            self._make_response(html, url="https://example.com/page/").next_page_url()
            is None
        )

    def test_fragment_only_is_rejected(self):
        html = '<html><body><a rel="next" href="#top">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_fragment_in_url_is_rejected(self):
        html = '<html><body><a rel="next" href="/page/2#section">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_invalid_scheme_javascript(self):
        html = '<html><body><a rel="next" href="javascript:alert(1)">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_invalid_scheme_mailto(self):
        html = '<html><body><a rel="next" href="mailto:x@example.com">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_invalid_scheme_ftp(self):
        html = '<html><body><a rel="next" href="ftp://files.example.com/">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_empty_href_is_skipped(self):
        html = '<html><body><a rel="next" href="">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_missing_href_is_skipped(self):
        html = '<html><body><a rel="next">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    # --- External origins (same-origin restriction) ---

    def test_external_origin_is_rejected(self):
        html = '<html><body><a rel="next" href="https://other.com/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_cross_origin_rel_next_is_rejected(self):
        html = '<html><body><a rel="next" href="//other.com/page/2">Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_https_explicit_port_443_is_same_origin(self):
        html = '<html><body><a rel="next" href="https://example.com:443/page/2">Next</a></body></html>'
        # urljoin preserves the explicit port; same-origin check normalizes 443 for https.
        assert (
            self._make_response(html, url="https://example.com/page/1").next_page_url()
            == "https://example.com:443/page/2"
        )

    def test_https_port_8443_is_different_origin(self):
        html = '<html><body><a rel="next" href="https://example.com:8443/page/2">Next</a></body></html>'
        assert self._make_response(html, url="https://example.com/page/1").next_page_url() is None

    def test_http_vs_https_is_different_origin(self):
        html = '<html><body><a rel="next" href="http://example.com/page/2">Next</a></body></html>'
        assert self._make_response(html, url="https://example.com/page/1").next_page_url() is None

    # --- aria-labelledby resolution ---

    def test_aria_labelledby_resolves_referenced_element_text(self):
        html = (
            '<html><body>'
            '<span id="next-label">Next page</span>'
            '<a aria-labelledby="next-label" href="/page/2">Next</a>'
            "</body></html>"
        )
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_aria_labelledby_case_insensitive_signal(self):
        html = (
            '<html><body>'
            '<span id="lbl">NEXT</span>'
            '<a aria-labelledby="lbl" href="/page/2">Next</a>'
            "</body></html>"
        )
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    def test_aria_labelledby_unresolved_id_is_skipped(self):
        html = '<html><body><a aria-labelledby="nonexistent-id" href="/page/2">Click here</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_aria_label_takes_precedence_over_labelledby(self):
        html = (
            '<html><body>'
            '<span id="lbl">Prev</span>'
            '<a aria-label="Next" aria-labelledby="lbl" href="/page/2">Next</a>'
            "</body></html>"
        )
        assert self._make_response(html).next_page_url() == "https://example.com/page/2"

    # --- False positives / missing ---

    def test_no_pagination_returns_none(self):
        html = '<html><body><a href="/about">About</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    def test_empty_html_returns_none(self):
        assert self._make_response("<html><body></body></html>").next_page_url() is None

    def test_text_only_next_without_href_is_skipped(self):
        html = '<html><body><a>Next</a></body></html>'
        assert self._make_response(html).next_page_url() is None

    # --- Existing API compatibility ---

    def test_next_page_url_does_not_break_css(self):
        html = '<html><body><a rel="next" href="/page/2">Next</a></body></html>'
        response = self._make_response(html)
        assert response.next_page_url() == "https://example.com/page/2"
        assert response.css("a::attr(href)").get() == "/page/2"

    def test_next_page_url_does_not_break_follow(self):
        html = '<html><body><a rel="next" href="/page/2">Next</a></body></html>'
        response = self._make_response(html)
        url = response.next_page_url()
        assert url is not None
        # follow() still works on the response
        assert response.url == "https://example.com/page"


# Candidate links that urllib refuses to parse: an unbalanced IPv6 bracket makes ``urljoin()``
# raise ``ValueError``, and an invalid port makes reading ``.port`` raise it. Each href also
# carries a page number, so it qualifies for the numbered-pagination signal as well.
MALFORMED_NEXT_HREFS = [
    pytest.param("http://[::1/list?page=2", id="ipv6-bracket-not-closed"),
    pytest.param("https://example.com]/list?page=2", id="ipv6-bracket-not-opened"),
    pytest.param("//[example.com/list?page=2", id="scheme-relative-bracket-not-closed"),
    pytest.param("https://example.com:abc/list?page=2", id="port-not-numeric"),
    pytest.param("https://example.com:99999/list?page=2", id="port-out-of-range"),
]

_BODY_PAGE = "<html><head></head><body>{links}</body></html>"
_HEAD_PAGE = "<html><head>{links}</head><body></body></html>"
# ``aria-labelledby`` is resolved against the document's top-level elements, so <body> carries the label id.
_LABELLED_PAGE = '<html><head></head><body id="next-label">Next page {links}</body></html>'

_REL_NEXT_LINK = '<a rel="next" href="{href}">go</a>'
_ARIA_LABEL_LINK = '<a aria-label="Next page" href="{href}">go</a>'
_ARIA_LABELLEDBY_LINK = '<a aria-labelledby="next-label" href="{href}">go</a>'
_TEXT_LINK = '<a href="{href}">Next</a>'

_PLAIN_URL = "https://example.com/list"  # No page number: the numbered-pagination signal stays out of the way
_NUMBERED_URL = "https://example.com/list?page=1"

# One entry per supported way of marking a next-page link: (page template, link template, response URL).
# Except for the link-text entry, the link text is never a next-page keyword, so only the named signal can match.
NEXT_PAGE_SIGNALS = [
    pytest.param(_HEAD_PAGE, '<link rel="next" href="{href}">', _PLAIN_URL, id="rel-next-on-link"),
    pytest.param(_BODY_PAGE, _REL_NEXT_LINK, _PLAIN_URL, id="rel-next-on-a"),
    pytest.param(_BODY_PAGE, '<map><area rel="next" href="{href}"></map>', _PLAIN_URL, id="rel-next-on-area"),
    pytest.param(_BODY_PAGE, _ARIA_LABEL_LINK, _PLAIN_URL, id="aria-label"),
    pytest.param(_LABELLED_PAGE, _ARIA_LABELLEDBY_LINK, _PLAIN_URL, id="aria-labelledby"),
    pytest.param(_BODY_PAGE, _TEXT_LINK, _PLAIN_URL, id="link-text"),
    pytest.param(_BODY_PAGE, '<a href="{href}">2</a>', _NUMBERED_URL, id="page-number"),
]


class TestNextPageUrlMalformedCandidates:
    """Response.next_page_url() skips candidate links whose URL cannot be parsed instead of raising."""

    _make_response = staticmethod(TestNextPageUrl._make_response)

    @classmethod
    def _detect(cls, page: str, link: str, url: str, *hrefs: str):
        html = page.format(links="".join(link.format(href=href) for href in hrefs))
        return cls._make_response(html, url=url).next_page_url()

    # --- Every signal: the malformed candidate is skipped ---

    @pytest.mark.parametrize("page,link,url", NEXT_PAGE_SIGNALS)
    def test_signal_detects_a_well_formed_link(self, page, link, url):
        # Guards the tests below against passing only because the signal never fired
        assert self._detect(page, link, url, "/list?page=3") == "https://example.com/list?page=3"

    @pytest.mark.parametrize("href", MALFORMED_NEXT_HREFS)
    @pytest.mark.parametrize("page,link,url", NEXT_PAGE_SIGNALS)
    def test_malformed_candidate_is_skipped(self, page, link, url, href):
        assert self._detect(page, link, url, href) is None

    @pytest.mark.parametrize("href", MALFORMED_NEXT_HREFS)
    @pytest.mark.parametrize("page,link,url", NEXT_PAGE_SIGNALS)
    def test_valid_candidate_after_malformed_one_is_returned(self, page, link, url, href):
        result = self._detect(page, link, url, href, "/list?page=3")
        assert result == "https://example.com/list?page=3"
        assert type(result) is str

    # --- Detection precedence ---

    @pytest.mark.parametrize("href", MALFORMED_NEXT_HREFS)
    @pytest.mark.parametrize(
        "malformed_link,valid_link,url",
        [
            pytest.param(_REL_NEXT_LINK, _ARIA_LABEL_LINK, _PLAIN_URL, id="rel-next-to-aria-label"),
            pytest.param(_ARIA_LABEL_LINK, _TEXT_LINK, _PLAIN_URL, id="aria-label-to-link-text"),
            pytest.param(_TEXT_LINK, '<a href="{href}">3</a>', _NUMBERED_URL, id="link-text-to-page-number"),
        ],
    )
    def test_malformed_candidate_falls_through_to_the_next_signal(self, malformed_link, valid_link, url, href):
        html = _BODY_PAGE.format(links=malformed_link.format(href=href) + valid_link.format(href="/list?page=3"))
        assert self._make_response(html, url=url).next_page_url() == "https://example.com/list?page=3"

    @pytest.mark.parametrize("href", MALFORMED_NEXT_HREFS)
    def test_higher_priority_signal_still_wins_over_malformed_lower_ones(self, href):
        html = _BODY_PAGE.format(
            links=(
                f'<a href="{href}">2</a>'
                + _TEXT_LINK.format(href=href)
                + _ARIA_LABEL_LINK.format(href=href)
                + _REL_NEXT_LINK.format(href="/list?page=3")
            )
        )
        assert self._make_response(html, url=_NUMBERED_URL).next_page_url() == "https://example.com/list?page=3"

    @pytest.mark.parametrize("href", MALFORMED_NEXT_HREFS)
    def test_first_valid_candidate_in_document_order_wins_around_malformed_ones(self, href):
        hrefs = (href, "/list?page=3", href, "/list?page=4")
        assert self._detect(_BODY_PAGE, _TEXT_LINK, _PLAIN_URL, *hrefs) == "https://example.com/list?page=3"

    # --- The other rejections still apply ---

    @pytest.mark.parametrize("href", MALFORMED_NEXT_HREFS)
    def test_other_rejections_still_apply_after_a_malformed_candidate(self, href):
        rejected = (
            href,
            "https://other.example/list?page=2",  # cross-origin
            "http://example.com/list?page=2",  # other scheme, so another origin
            "https://example.com:8443/list?page=2",  # other port, so another origin
            "https://example.com/list",  # self-link
            "/list/",  # self-link with a trailing slash
            "#next",  # fragment only
            "/list?page=2#top",  # URL with a fragment
            "javascript:void(0)",
            "mailto:someone@example.com",
            "ftp://example.com/list?page=2",
        )
        assert self._detect(_BODY_PAGE, _TEXT_LINK, _PLAIN_URL, *rejected) is None
        detected = self._detect(_BODY_PAGE, _TEXT_LINK, _PLAIN_URL, *rejected, "/list?page=3")
        assert detected == "https://example.com/list?page=3"

    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("https://example.com:abc/list?page=1", id="port-not-numeric"),
            pytest.param("http://[::1/list?page=1", id="ipv6-bracket-not-closed"),
        ],
    )
    def test_response_url_that_cannot_be_parsed_yields_none(self, url):
        # Without a usable origin for the page itself, no candidate can be shown to be same-origin
        links = (
            _REL_NEXT_LINK.format(href="/list?page=2")
            + _REL_NEXT_LINK.format(href="https://example.com/list?page=2")
            + '<a href="?page=3">3</a>'
        )
        assert self._make_response(_BODY_PAGE.format(links=links), url=url).next_page_url() is None

    def test_well_formed_ipv6_and_port_links_are_still_detected(self):
        url = "http://[::1]:8080/list?page=1"
        for href in ("/list?page=2", "http://[::1]:8080/list?page=2"):
            assert self._detect(_BODY_PAGE, _REL_NEXT_LINK, url, href) == "http://[::1]:8080/list?page=2"
        assert self._detect(_BODY_PAGE, _REL_NEXT_LINK, url, "http://[::1]:9090/list?page=2") is None
