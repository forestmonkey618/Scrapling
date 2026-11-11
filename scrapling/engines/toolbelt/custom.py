"""
Functions related to custom types or type checking
"""

import re
from urllib.parse import urlparse
from functools import lru_cache

from scrapling.core.utils import log
from scrapling.core._types import (
    Any,
    Dict,
    cast,
    List,
    Tuple,
    Union,
    Optional,
    Callable,
    Sequence,
    TYPE_CHECKING,
    AsyncGenerator,
)
from scrapling.core.custom_types import MappingProxyType
from scrapling.parser import Selector, SQLiteStorageSystem

if TYPE_CHECKING:
    from scrapling.spiders import Request


_NON_INHERITABLE_KWARGS = frozenset({"method", "data", "json", "params", "files", "multipart"})

# Signals used by Response.next_page_url(), checked in strict precedence order.
# Priority 1 (rel="next") is handled by attribute inspection; the text/aria sets
# below are used for priorities 2 and 3.
_NEXT_PAGE_TEXT_SIGNALS = frozenset({
    "next", "›", ">", "»", "next page", "older", "newer",
})

# Common URL query/path patterns that indicate a numbered page.
_PAGE_NUMBER_URL_PATTERNS = (
    re.compile(r"[?&]page=(\d+)(?:&|$)", re.IGNORECASE),
    re.compile(r"[?&]p=(\d+)(?:&|$)", re.IGNORECASE),
    re.compile(r"/page/(\d+)/", re.IGNORECASE),
)


def _extract_page_number(url: str) -> Optional[int]:
    """Extract a page number from a URL, or ``None`` if no page pattern is found."""
    for pattern in _PAGE_NUMBER_URL_PATTERNS:
        match = pattern.search(url)
        if match:
            return int(match.group(1))
    return None


def _origin_key(url: str) -> tuple:
    """Return a comparable (scheme, hostname, port) tuple for origin comparison.

    Default ports (80 for http, 443 for https) are normalized so that
    ``http://example.com/`` and ``http://example.com:80/`` compare equal.
    """
    parsed = urlparse(url)
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    return (parsed.scheme.lower(), parsed.hostname.lower() if parsed.hostname else "", port)


class Response(Selector):
    """This class is returned by all engines as a way to unify the response type between different libraries.

    :param status: HTTP status code.
    :param reason: HTTP status message.
    :param cookies: Response cookies.
    :param headers: Response headers.
    :param request_headers: Request headers sent with the request.
    :param history: List of redirect responses, if any.
    :param meta: Metadata dictionary (e.g., proxy used).
    :param request: Associated spider Request object (set by crawler, in the spiders framework).
    """

    def __init__(
        self,
        url: str,
        content: str | bytes,
        status: int,
        reason: str,
        cookies: Tuple[Dict[str, str], ...] | Dict[str, str],
        headers: Dict,
        request_headers: Dict,
        encoding: str = "utf-8",
        method: str = "GET",
        history: List | None = None,
        meta: Dict[str, Any] | None = None,
        *,
        _log: bool = True,
        **selector_config: Any,
    ):
        if isinstance(content, str):
            content = content.encode("utf-8")

        adaptive_domain: str = cast(str, selector_config.pop("adaptive_domain", ""))
        self.status = status
        self.method = method
        self.reason = reason
        self.cookies = cookies
        self.headers = headers
        self.request_headers = request_headers
        self.history = history or []
        super().__init__(
            content=content,
            url=adaptive_domain or url,
            encoding=encoding,
            **selector_config,
        )
        if _log:
            log.info(f"Fetched ({status}) <{method} {url}> (referer: {request_headers.get('referer')})")

        if meta and not isinstance(meta, dict):
            raise TypeError(f"Response meta should be dictionary but got {type(meta).__name__} instead!")

        self.meta: Dict[str, Any] = meta or {}
        self.request: Optional["Request"] = None  # Will be set by crawler

    @property
    def body(self) -> bytes:
        """Return the raw body of the response as bytes."""
        return cast(bytes, cast(Sequence, self._raw_body))

    def markdown(self, css_selector: Optional[str] = None, main_content_only: bool = False) -> str:
        """Convert the response content to clean Markdown.

        Scripts, styles, and hidden/prompt-injection content are always removed before conversion, which is
        the same cleaning the MCP server does. Requires the "markdownify" package (`pip install "scrapling[rag]"`).

        :param css_selector: CSS selector to convert only the matching elements. All matches are concatenated.
        :param main_content_only: Convert only the content inside the `<body>` tag.
        """
        from scrapling.core.shell import Convertor

        page = (cast(Selector, self.css("body").first) or self) if main_content_only else self
        page = Convertor._sanitize_for_ai(Convertor._strip_noise_tags(page))
        pages = [page] if not css_selector else page.css(css_selector)
        return "\n\n".join(Convertor._convert_to_markdown(element.html_content) for element in pages)

    def follow(
        self,
        url: str,
        sid: str = "",
        callback: Callable[["Response"], AsyncGenerator[Union[Dict[str, Any], "Request", None], None]] | None = None,
        priority: int | None = None,
        dont_filter: bool = False,
        meta: dict[str, Any] | None = None,
        referer_flow: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Create a Request to follow a URL.

        This is a helper method for spiders to easily follow links found in pages.

        **IMPORTANT**: The below arguments if left empty, the corresponding value from the previous request will be used. The only exception is `dont_filter`.

        The session kwargs of the previous request are inherited as well, except `method`, `params`, `data`, `json`, `files`, and `multipart`, so a followed link is a plain GET unless you pass them again.

        :param url: The URL to follow (can be relative, will be joined with current URL)
        :param sid: The session id to use
        :param callback: Spider callback method to use
        :param priority: The priority number to use, the higher the number, the higher priority to be processed first.
        :param dont_filter: If this request has been done before, disable the filter to allow it again.
        :param meta: Additional meta data to included in the request
        :param referer_flow: Enabled by default, set the current response url as referer for the new request url.
        :param kwargs: Additional Request arguments
        :return: Request object ready to be yielded
        """
        from scrapling.spiders import Request

        if not self.request or not isinstance(self.request, Request):
            raise TypeError("This response has no request set yet.")

        # Merge original session kwargs with new kwargs (new takes precedence)
        inherited = {k: v for k, v in self.request._session_kwargs.items() if k not in _NON_INHERITABLE_KWARGS}
        for key in ("headers", "extra_headers"):
            if inherited.get(key):
                inherited[key] = {k: v for k, v in inherited[key].items() if k.lower() != "content-length"}
        session_kwargs = {**inherited, **kwargs}

        if referer_flow:
            # For requests
            headers = session_kwargs.get("headers", {})
            headers["referer"] = self.url
            session_kwargs["headers"] = headers

            # For browsers
            extra_headers = session_kwargs.get("extra_headers", {})
            extra_headers["referer"] = self.url
            session_kwargs["extra_headers"] = extra_headers

            session_kwargs["google_search"] = False

        return Request(
            url=self.urljoin(url),
            sid=sid or self.request.sid,
            callback=callback or self.request.callback,
            priority=priority if priority is not None else self.request.priority,
            dont_filter=dont_filter,
            meta={**(self.meta or {}), **(meta or {})},
            **session_kwargs,
        )

    def next_page_url(self) -> Optional[str]:
        """Return the absolute URL of the next page, or ``None`` if none is detected.

        Detection uses deterministic signals in precedence order (first match wins):

        1. ``rel="next"`` on ``<link>``, ``<a>``, or ``<area>`` elements — multi-token
           and case-insensitive (e.g. ``rel="NEXT"``, ``rel="next archive"``).
        2. Accessibility labels (``aria-label`` / ``aria-labelledby``) containing a
           next-page keyword.
        3. Link text matching a next-page keyword ("Next", "›", ">", "»", "next page",
           "older", "newer").
        4. Numbered pagination URLs whose page number exceeds the current page's number.

        Within each priority level, the first candidate in document order wins.

        The method only *detects* — it does not navigate, issue network requests, or
        modify crawler state. Use :meth:`follow` to navigate to the returned URL.

        :return: Absolute next-page URL, or ``None`` if no next page is detected.
        """
        current_url = self.url
        seen: set[str] = set()

        for url in self._iter_rel_next_candidates():
            resolved = self._validate_next_url(url, seen)
            if resolved:
                return resolved

        for url in self._iter_aria_label_candidates():
            resolved = self._validate_next_url(url, seen)
            if resolved:
                return resolved

        for url in self._iter_text_candidates():
            resolved = self._validate_next_url(url, seen)
            if resolved:
                return resolved

        for url in self._iter_page_number_candidates():
            resolved = self._validate_next_url(url, seen)
            if resolved:
                return resolved

        return None

    def _iter_rel_next_candidates(self):
        """Yield href URLs from elements with rel containing 'next' (case-insensitive,
        multi-token). Priority 1 signal."""
        for element in self.css("a[rel], area[rel], link[rel]"):
            rel = element.attrib.get("rel")
            if not rel:
                continue
            tokens = str(rel).lower().split()
            if "next" in tokens:
                href = element.attrib.get("href")
                if href:
                    yield str(href)

    def _iter_aria_label_candidates(self):
        """Yield href URLs from elements with a next-page aria-label/aria-labelledby.
        Priority 2 signal.

        ``aria-label`` is used directly as the label text. ``aria-labelledby`` is an
        ID reference whose referenced element's text content is resolved and used as
        the label (per the ARIA spec).
        """
        for element in self.css(
            "a[aria-label], a[aria-labelledby], area[aria-label], area[aria-labelledby]"
        ):
            aria_label = element.attrib.get("aria-label")
            if aria_label:
                label_text = str(aria_label)
            else:
                labelledby = element.attrib.get("aria-labelledby")
                if not labelledby:
                    continue
                label_text = self._resolve_aria_labelledby(labelledby)
            if not label_text:
                continue
            label_lower = label_text.lower()
            if any(signal in label_lower for signal in _NEXT_PAGE_TEXT_SIGNALS):
                href = element.attrib.get("href")
                if href:
                    yield str(href)

    def _resolve_aria_labelledby(self, label_id_ref: str) -> str:
        """Resolve aria-labelledby ID reference(s) to concatenated element text.

        ``aria-labelledby`` contains space-separated element IDs. Each referenced
        element's text content is looked up and joined with spaces.
        """
        parts: list[str] = []
        for ref_id in str(label_id_ref).split():
            ref_id = ref_id.strip()
            if not ref_id:
                continue
            # Use XPath variable binding to avoid CSS injection in the ID reference.
            for ref_element in self.xpath("*[@id=$ref_id]", ref_id=ref_id):
                text = str(ref_element.get_all_text(strip=True))
                if text:
                    parts.append(text)
                    break
        return " ".join(parts)

    def _iter_text_candidates(self):
        """Yield href URLs from <a> elements whose text matches a next-page keyword.
        Priority 3 signal."""
        for element in self.css("a"):
            text = str(element.get_all_text(strip=True)).lower()
            if not text:
                continue
            if any(text == signal or text.startswith(signal + " ") for signal in _NEXT_PAGE_TEXT_SIGNALS):
                href = element.attrib.get("href")
                if href:
                    yield str(href)

    def _iter_page_number_candidates(self):
        """Yield href URLs whose page number exceeds the current page's number.
        Priority 4 (conservative) signal."""
        current_page = _extract_page_number(self.url)
        if current_page is None:
            return
        for element in self.css("a[href]"):
            href = str(element.attrib.get("href"))
            page_num = _extract_page_number(href)
            if page_num is not None and page_num > current_page:
                yield href

    def _validate_next_url(self, href: str, seen: set[str]) -> Optional[str]:
        """Resolve and validate a candidate next-page URL.

        Returns the absolute URL if valid, or ``None`` if it should be skipped.
        """
        current_url = self.url

        if not href or not href.strip():
            return None

        href = href.strip()

        # Reject fragment-only navigation.
        if href == "#" or href.startswith("#"):
            return None

        # Resolve relative URLs against the response URL.
        absolute = str(self.urljoin(href))

        # Reject empty or fragment-containing results.
        if not absolute or "#" in absolute:
            return None

        # Accept HTTP/HTTPS only.
        lower = absolute.lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            return None

        # Reject cross-origin URLs (same-origin restriction).
        if _origin_key(absolute) != _origin_key(current_url):
            return None

        # Reject self-links (ignore trailing slash and bare query differences).
        if absolute.rstrip("/?") == current_url.rstrip("/?"):
            return None

        # Reject duplicates.
        if absolute in seen:
            return None
        seen.add(absolute)

        return absolute

    def __str__(self) -> str:
        return f"<{self.status} {self.url}>"


class BaseFetcher:
    __slots__ = ()
    huge_tree: bool = True
    adaptive: Optional[bool] = False
    storage: Any = SQLiteStorageSystem
    storage_args: Optional[Dict] = None
    keep_comments: Optional[bool] = False
    adaptive_domain: str = ""
    parser_keywords: Tuple = (
        "huge_tree",
        "adaptive",
        "storage",
        "storage_args",
        "keep_comments",
        "adaptive_domain",
    )  # Left open for the user

    def __init__(self, *args, **kwargs):
        # For backward-compatibility before 0.2.99
        args_str = ", ".join(args) or ""
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items()) or ""
        if args_str:
            args_str += ", "

        log.warning(
            f"This logic is deprecated now, and have no effect; It will be removed with v0.3. Use `{self.__class__.__name__}.configure({args_str}{kwargs_str})` instead before fetching"
        )
        pass

    @classmethod
    def display_config(cls):
        return dict(
            huge_tree=cls.huge_tree,
            keep_comments=cls.keep_comments,
            adaptive=cls.adaptive,
            storage=cls.storage,
            storage_args=cls.storage_args,
            adaptive_domain=cls.adaptive_domain,
        )

    @classmethod
    def configure(cls, **kwargs):
        """Set multiple arguments for the parser at once globally

        :param kwargs: The keywords can be any arguments of the following: huge_tree, keep_comments, adaptive, storage, storage_args, adaptive_domain
        """
        for key, value in kwargs.items():
            key = key.strip().lower()
            if hasattr(cls, key):
                if key in cls.parser_keywords:
                    setattr(cls, key, value)
                else:
                    # Yup, no fun allowed LOL
                    raise AttributeError(f'Unknown parser argument: "{key}"; maybe you meant {cls.parser_keywords}?')
            else:
                raise ValueError(f'Unknown parser argument: "{key}"; maybe you meant {cls.parser_keywords}?')

        if not kwargs:
            raise AttributeError(f"You must pass a keyword to configure, current keywords: {cls.parser_keywords}?")

    @classmethod
    def _generate_parser_arguments(cls) -> Dict:
        # Selector class parameters
        # I won't validate Selector's class parameters here again, I will leave it to be validated later
        parser_arguments = dict(
            huge_tree=cls.huge_tree,
            keep_comments=cls.keep_comments,
            adaptive=cls.adaptive,
            storage=cls.storage,
            storage_args=cls.storage_args,
            adaptive_domain=cls.adaptive_domain,
        )

        return parser_arguments


class StatusText:
    """A class that gets the status text of the response status code.

    Reference: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    """

    _phrases = MappingProxyType(
        {
            100: "Continue",
            101: "Switching Protocols",
            102: "Processing",
            103: "Early Hints",
            200: "OK",
            201: "Created",
            202: "Accepted",
            203: "Non-Authoritative Information",
            204: "No Content",
            205: "Reset Content",
            206: "Partial Content",
            207: "Multi-Status",
            208: "Already Reported",
            226: "IM Used",
            300: "Multiple Choices",
            301: "Moved Permanently",
            302: "Found",
            303: "See Other",
            304: "Not Modified",
            305: "Use Proxy",
            307: "Temporary Redirect",
            308: "Permanent Redirect",
            400: "Bad Request",
            401: "Unauthorized",
            402: "Payment Required",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            406: "Not Acceptable",
            407: "Proxy Authentication Required",
            408: "Request Timeout",
            409: "Conflict",
            410: "Gone",
            411: "Length Required",
            412: "Precondition Failed",
            413: "Payload Too Large",
            414: "URI Too Long",
            415: "Unsupported Media Type",
            416: "Range Not Satisfiable",
            417: "Expectation Failed",
            418: "I'm a teapot",
            421: "Misdirected Request",
            422: "Unprocessable Entity",
            423: "Locked",
            424: "Failed Dependency",
            425: "Too Early",
            426: "Upgrade Required",
            428: "Precondition Required",
            429: "Too Many Requests",
            431: "Request Header Fields Too Large",
            451: "Unavailable For Legal Reasons",
            500: "Internal Server Error",
            501: "Not Implemented",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
            505: "HTTP Version Not Supported",
            506: "Variant Also Negotiates",
            507: "Insufficient Storage",
            508: "Loop Detected",
            510: "Not Extended",
            511: "Network Authentication Required",
        }
    )

    @classmethod
    @lru_cache(maxsize=128)
    def get(cls, status_code: int) -> str:
        """Get the phrase for a given HTTP status code."""
        return cls._phrases.get(status_code, "Unknown Status Code")
