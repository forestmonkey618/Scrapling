"""Tests for the shared template helper `_decompress`."""

import gzip

import pytest

from scrapling.spiders.templates import _utils
from scrapling.spiders.templates._utils import _decompress


PAYLOAD = b"<rss><channel><title>Feed Title</title></channel></rss>"


def _damaged(body: bytes, damage: str) -> bytes:
    """Drop the gzip trailer or corrupt the deflate stream, keeping the gzip header readable."""
    return body[:-4] if damage == "truncated" else body[:10] + b"\x07" + body[11:]


class TestDecompressValidInput:
    def test_gzip_magic_bytes_are_decompressed(self):
        assert _decompress(gzip.compress(PAYLOAD), None) == PAYLOAD

    def test_gzip_content_type_is_decompressed(self):
        assert _decompress(gzip.compress(PAYLOAD), "application/gzip") == PAYLOAD

    def test_content_type_match_is_case_insensitive(self):
        assert _decompress(gzip.compress(PAYLOAD), "Application/X-GZIP") == PAYLOAD

    def test_multi_member_gzip_is_fully_decompressed(self):
        body = gzip.compress(b"first") + gzip.compress(b"second")
        assert _decompress(body, None) == b"firstsecond"

    def test_plain_body_is_returned_unchanged(self):
        assert _decompress(PAYLOAD, "application/xml") == PAYLOAD

    def test_empty_body_is_returned_unchanged(self):
        assert _decompress(b"", None) == b""


class TestDecompressMalformedInput:
    """Feed and sitemap templates catch `OSError`, so malformed gzip must stay inside that family."""

    @pytest.mark.parametrize("damage", ["truncated", "deflate"])
    @pytest.mark.parametrize("content_type", [None, "application/gzip"])
    def test_malformed_gzip_raises_bad_gzip_file(self, damage, content_type):
        body = _damaged(gzip.compress(PAYLOAD), damage)

        with pytest.raises(gzip.BadGzipFile):
            _decompress(body, content_type)

    @pytest.mark.parametrize("damage", ["truncated", "deflate"])
    def test_malformed_gzip_is_catchable_as_oserror(self, damage):
        body = _damaged(gzip.compress(PAYLOAD), damage)

        with pytest.raises(OSError):
            _decompress(body, None)

    def test_truncated_gzip_keeps_the_original_exception_as_cause(self):
        with pytest.raises(gzip.BadGzipFile) as excinfo:
            _decompress(gzip.compress(PAYLOAD)[:-4], None)

        assert isinstance(excinfo.value.__cause__, EOFError)
        assert "end-of-stream" in str(excinfo.value)

    def test_corrupt_deflate_keeps_the_original_exception_as_cause(self):
        with pytest.raises(gzip.BadGzipFile) as excinfo:
            _decompress(_damaged(gzip.compress(PAYLOAD), "deflate"), None)

        assert excinfo.value.__cause__ is not None
        assert "decompressing data" in str(excinfo.value)

    def test_non_gzip_body_declared_as_gzip_still_raises_bad_gzip_file(self):
        with pytest.raises(gzip.BadGzipFile):
            _decompress(b"this is not gzip at all", "application/gzip")


class TestDecompressSizeLimit:
    def test_default_limit_is_64_mib(self):
        assert _utils._GUNZIP_MAX_SIZE == 64 * 1024 * 1024

    def test_output_above_the_limit_raises_oserror(self, monkeypatch):
        monkeypatch.setattr(_utils, "_GUNZIP_MAX_SIZE", 16)

        with pytest.raises(OSError, match="gzip output exceeds 16 bytes"):
            _decompress(gzip.compress(b"x" * 64), None)

    def test_output_exactly_at_the_limit_is_allowed(self, monkeypatch):
        monkeypatch.setattr(_utils, "_GUNZIP_MAX_SIZE", 64)

        assert _decompress(gzip.compress(b"x" * 64), None) == b"x" * 64

    def test_size_limit_error_is_not_wrapped_as_bad_gzip_file(self, monkeypatch):
        """The cap raises a plain OSError; only EOFError/zlib.error are normalized."""
        monkeypatch.setattr(_utils, "_GUNZIP_MAX_SIZE", 16)

        with pytest.raises(OSError) as excinfo:
            _decompress(gzip.compress(b"x" * 64), None)

        assert not isinstance(excinfo.value, gzip.BadGzipFile)

    def test_real_64_mib_cap_is_enforced(self):
        payload = b"\x00" * (_utils._GUNZIP_MAX_SIZE + 1)

        with pytest.raises(OSError, match=f"gzip output exceeds {_utils._GUNZIP_MAX_SIZE} bytes"):
            _decompress(gzip.compress(payload, compresslevel=1), None)
