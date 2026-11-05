import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from unittest.mock import patch, MagicMock
from curl_cffi.curl import CurlError
from curl_cffi.const import CurlFollow

from scrapling.engines.static import _SyncSessionLogic as FetcherSession, FetcherClient
from scrapling.engines.toolbelt import ProxyRotator


class TestFetcherSession:
    """Test FetcherSession functionality"""

    def test_fetcher_session_creation(self):
        """Test FetcherSession creation"""
        session = FetcherSession(timeout=30, retries=3, stealthy_headers=True)

        assert session._default_timeout == 30
        assert session._default_retries == 3

    def test_fetcher_session_context_manager(self):
        """Test FetcherSession as a context manager"""
        session = FetcherSession()

        with session as s:
            assert s == session
            assert session._curl_session is not None

        # Session should be cleaned up

    def test_fetcher_session_double_enter(self):
        """Test error on double entering"""
        session = FetcherSession()

        with session:
            with pytest.raises(RuntimeError):
                session.__enter__()

    def test_fetcher_client_creation(self):
        """Test FetcherClient creation"""
        client = FetcherClient()

        # Should not have context manager methods
        assert client.__enter__ is None
        assert client.__exit__ is None

    def test_session_level_proxy_is_applied(self):
        """Session-level proxy must reach the request, not be silently dropped (#295)"""
        proxy = "http://10.255.255.1:9999"

        with FetcherSession(proxy=proxy) as session:
            with (
                patch.object(session._curl_session, "request") as mocked_request,
                patch("scrapling.engines.static.ResponseFactory.from_http_request", return_value=MagicMock()),
            ):
                session.get("http://example.com")

            assert mocked_request.call_args.kwargs["proxy"] == proxy

    def test_per_request_proxy_overrides_session_proxy(self):
        """A per-request proxy must take precedence over the session-level proxy"""
        request_proxy = "http://10.255.255.2:9999"

        with FetcherSession(proxy="http://10.255.255.1:9999") as session:
            with (
                patch.object(session._curl_session, "request") as mocked_request,
                patch("scrapling.engines.static.ResponseFactory.from_http_request", return_value=MagicMock()),
            ):
                session.get("http://example.com", proxy=request_proxy)

            assert mocked_request.call_args.kwargs["proxy"] == request_proxy

    def test_proxy_rotates_per_retry_attempt(self):
        """With a rotator, every retry attempt must pull a fresh proxy"""
        rotator = ProxyRotator(["http://p1:8080", "http://p2:8080"])

        with FetcherSession(proxy_rotator=rotator, retries=2, retry_delay=0) as session:
            with (
                patch.object(session._curl_session, "request") as mocked_request,
                patch("scrapling.engines.static.ResponseFactory.from_http_request", return_value=MagicMock()),
            ):
                mocked_request.side_effect = [CurlError("transient"), MagicMock()]
                session.get("http://example.com")

            proxies_used = [call.kwargs["proxy"] for call in mocked_request.call_args_list]
            assert proxies_used == ["http://p1:8080", "http://p2:8080"]

    @pytest.mark.parametrize("retries", [0, -1, None])
    def test_retries_below_one_still_sends_the_request(self, retries):
        """A session-level retries below 1 must still send the request once instead of skipping it"""
        with FetcherSession(retries=retries, retry_delay=0) as session:
            with (
                patch.object(session._curl_session, "request") as mocked_request,
                patch("scrapling.engines.static.ResponseFactory.from_http_request", return_value=MagicMock()),
            ):
                session.get("http://example.com")

            assert mocked_request.call_count == 1

    def test_per_request_retries_below_one_still_sends_the_request(self):
        """A per-request retries of 0 must override the session default without skipping the request"""
        with FetcherSession(retries=3, retry_delay=0) as session:
            with (
                patch.object(session._curl_session, "request") as mocked_request,
                patch("scrapling.engines.static.ResponseFactory.from_http_request", return_value=MagicMock()),
            ):
                session.get("http://example.com", retries=0)

            assert mocked_request.call_count == 1

    def test_request_headers_replace_session_headers_regardless_of_case(self):
        """The server must receive one value per header, the per-request one, whatever the case"""

        class EchoHeaders(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(
                    {
                        "user_agent": self.headers.get_all("User-Agent") or [],
                        "authorization": self.headers.get_all("Authorization") or [],
                        "accept": self.headers.get_all("Accept") or [],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), EchoHeaders)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        session_headers = {"User-Agent": "SessionUA/1.0", "Authorization": "Bearer SESSION", "Accept": "text/plain"}
        try:
            with FetcherSession(headers=session_headers) as session:
                lowercase = session.get(url, headers={"user-agent": "RequestUA/2.0", "authorization": "Bearer REQUEST"})
                same_case = session.get(url, headers={"User-Agent": "RequestUA/2.0"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        assert lowercase.json() == {
            "user_agent": ["RequestUA/2.0"],
            "authorization": ["Bearer REQUEST"],
            "accept": ["text/plain"],
        }
        assert same_case.json() == {
            "user_agent": ["RequestUA/2.0"],
            "authorization": ["Bearer SESSION"],
            "accept": ["text/plain"],
        }


class TestFollowRedirects:
    """Test follow_redirects string normalization to curl_cffi CurlFollow enum."""

    def _capture_allow_redirects(self, session, follow_redirects=None, **kwargs):
        """Helper: capture the allow_redirects kwarg passed to curl_cffi session.request."""
        with (
            patch.object(session._curl_session, "request") as mocked_request,
            patch("scrapling.engines.static.ResponseFactory.from_http_request", return_value=MagicMock()),
        ):
            mocked_request.return_value = MagicMock()
            get_kwargs = dict(kwargs)
            if follow_redirects is not None:
                get_kwargs["follow_redirects"] = follow_redirects
            session.get("http://example.com", **get_kwargs)
            return mocked_request.call_args.kwargs.get("allow_redirects")

    @pytest.mark.parametrize(
        "follow_redirects,expected",
        [
            ("all", CurlFollow.ALL),
            ("obeycode", CurlFollow.OBEYCODE),
            ("firstonly", CurlFollow.FIRSTONLY),
            ("safe", CurlFollow.SAFE),
        ],
    )
    def test_string_follow_redirects_normalized_to_enum(self, follow_redirects, expected):
        """String follow_redirects values must be converted to CurlFollow enum members."""
        with FetcherSession(retries=1, retry_delay=0) as session:
            allow_redirects = self._capture_allow_redirects(session, follow_redirects)
            assert allow_redirects == expected, (
                f"follow_redirects={follow_redirects!r} should normalize to {expected}, got {allow_redirects!r}"
            )

    @pytest.mark.parametrize("follow_redirects", [True, False])
    def test_boolean_follow_redirects_preserved(self, follow_redirects):
        """Boolean follow_redirects values must pass through unchanged."""
        with FetcherSession(retries=1, retry_delay=0) as session:
            allow_redirects = self._capture_allow_redirects(session, follow_redirects)
            assert allow_redirects == follow_redirects, (
                f"follow_redirects={follow_redirects!r} should be preserved, got {allow_redirects!r}"
            )

    def test_session_default_follow_redirects_applied(self):
        """Session-level follow_redirects default must be normalized and applied."""
        with FetcherSession(follow_redirects="all", retries=1, retry_delay=0) as session:
            allow_redirects = self._capture_allow_redirects(session, None)
            assert allow_redirects == CurlFollow.ALL

    def test_per_request_follow_redirects_overrides_session_default(self):
        """Per-request follow_redirects must override the session-level default."""
        with FetcherSession(follow_redirects="safe", retries=1, retry_delay=0) as session:
            allow_redirects = self._capture_allow_redirects(session, "all")
            assert allow_redirects == CurlFollow.ALL

    def test_invalid_string_follow_redirects_raises_valueerror(self):
        """Unsupported string values must still raise ValueError (not silently accepted)."""
        with FetcherSession(retries=1, retry_delay=0) as session:
            with pytest.raises(ValueError, match="Invalid follow_redirects value"):
                session.get("http://example.com", follow_redirects="invalid")

    def test_default_follow_redirects_is_safe(self):
        """The default follow_redirects must be CurlFollow.SAFE after normalization."""
        with FetcherSession(retries=1, retry_delay=0) as session:
            allow_redirects = self._capture_allow_redirects(session)
            assert allow_redirects == CurlFollow.SAFE

    def test_actual_redirect_follows_with_string_all(self, redirect_server):
        """follow_redirects='all' must actually follow an HTTP redirect (real curl_cffi handling)."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects="all")
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200

    def test_actual_redirect_follows_with_string_obeycode(self, redirect_server):
        """follow_redirects='obeycode' must actually follow an HTTP redirect."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects="obeycode")
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200

    def test_actual_redirect_follows_with_string_firstonly(self, redirect_server):
        """follow_redirects='firstonly' must actually follow an HTTP redirect."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects="firstonly")
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200

    def test_actual_redirect_follows_with_bool_true(self, redirect_server):
        """follow_redirects=True must actually follow an HTTP redirect."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects=True)
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200

    def test_actual_redirect_disabled_with_bool_false(self, redirect_server):
        """follow_redirects=False must NOT follow redirects."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects=False)
        assert resp.url == f"{BASE}/redirect-relative"
        assert resp.status == 302
        assert len(resp.history) == 0

    def test_actual_redirect_chain_with_string_all(self, redirect_server):
        """follow_redirects='all' must follow a multi-hop redirect chain."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-chain-1", follow_redirects="all")
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200
        assert len(resp.history) == 2
        assert resp.history[0].url == f"{BASE}/redirect-chain-1"
        assert resp.history[1].url == f"{BASE}/redirect-chain-2"

    def test_actual_redirect_firstonly_follows_single_redirect(self, redirect_server):
        """follow_redirects='firstonly' must follow a single redirect (works like ALL for one hop)."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects="firstonly")
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200

    def test_actual_redirect_obeycode_follows_standard_redirect(self, redirect_server):
        """follow_redirects='obeycode' must follow a standard 302 redirect (same as ALL)."""
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects="obeycode")
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200

    @pytest.mark.parametrize("follow_redirects", ["ALL", "OBEYCODE", "FirstOnly"])
    def test_case_insensitive_follow_redirects(self, redirect_server, follow_redirects):
        """follow_redirects string values must be case-insensitive.

        Note: "Safe" is excluded because it triggers SSRF protection for localhost redirects.
        """
        BASE = redirect_server

        with FetcherSession(retries=1, retry_delay=0) as session:
            resp = session.get(f"{BASE}/redirect-relative", follow_redirects=follow_redirects)
        assert resp.url == f"{BASE}/final"
        assert resp.status == 200
