from pathlib import Path

import anyio
import pytest

from scrapling.engines.toolbelt.custom import Response
from scrapling.spiders.cache import ResponseCacheManager


@pytest.mark.anyio
async def test_concurrent_writers_use_independent_temporary_files(tmp_path, monkeypatch, caplog):
    fingerprint = b"shared-request"
    ready = anyio.Event()
    first_ready = anyio.Event()
    paths = []
    original_replace = anyio.Path.replace

    async def synchronized_replace(source, target):
        paths.append(str(source))
        first_ready.set()
        if len(paths) == 2:
            ready.set()
        await ready.wait()
        return await original_replace(source, target)

    monkeypatch.setattr(anyio.Path, "replace", synchronized_replace)

    async def write(body):
        response = Response(
            url="https://example.com",
            content=body,
            status=200,
            reason="OK",
            encoding="utf-8",
            cookies={},
            headers={},
            request_headers={},
            method="GET",
        )
        await ResponseCacheManager(tmp_path).put(fingerprint, response)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(write, b"<html>first</html>")
            await first_ready.wait()
            tasks.start_soon(write, b"<html>second response</html>")

    assert "Failed to cache response" not in caplog.text
    assert len(set(paths)) == 2
    response = await ResponseCacheManager(tmp_path).get(fingerprint)
    assert response is not None
    assert response.body in (b"<html>first</html>", b"<html>second response</html>")
    assert list(Path(tmp_path).glob("*.tmp")) == []


@pytest.mark.anyio
async def test_failed_replacement_keeps_previous_response(tmp_path, monkeypatch):
    cache = ResponseCacheManager(tmp_path)
    fingerprint = b"existing-request"

    def response(body):
        return Response(
            url="https://example.com",
            content=body,
            status=200,
            reason="OK",
            encoding="utf-8",
            cookies={},
            headers={},
            request_headers={},
            method="GET",
        )

    await cache.put(fingerprint, response(b"<html>original</html>"))

    async def fail_replace(source, target):
        raise OSError("replacement unavailable")

    monkeypatch.setattr(anyio.Path, "replace", fail_replace)
    await cache.put(fingerprint, response(b"<html>replacement</html>"))
    restored = await cache.get(fingerprint)
    assert restored is not None
    assert restored.body == b"<html>original</html>"
    assert list(Path(tmp_path).glob("*.tmp")) == []


def _response(body):
    return Response(
        url="https://example.com/page",
        content=body,
        status=200,
        reason="OK",
        encoding="utf-8",
        cookies={},
        headers={},
        request_headers={},
    )


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["open", "write"])
async def test_failed_write_keeps_previous_response_and_removes_temp(tmp_path, monkeypatch, caplog, failure):
    cache = ResponseCacheManager(tmp_path)
    fingerprint = b"write-failure"
    await cache.put(fingerprint, _response(b"original"))
    original_bytes = (tmp_path / f"{fingerprint.hex()}.json").read_bytes()
    original_open = anyio.open_file
    original_write = anyio.AsyncFile.write

    async def fail_open(path, mode):
        if mode == "wb":
            raise OSError("injected open failure")
        return await original_open(path, mode)

    async def fail_write(file, data):
        await original_write(file, data[: len(data) // 2])
        raise OSError("injected partial write failure")

    if failure == "open":
        monkeypatch.setattr(anyio, "open_file", fail_open)
    else:
        monkeypatch.setattr(anyio.AsyncFile, "write", fail_write)
    await cache.put(fingerprint, _response(b"replacement"))
    restored = await cache.get(fingerprint)
    assert restored is not None and restored.body == b"original"
    assert (tmp_path / f"{fingerprint.hex()}.json").read_bytes() == original_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"{fingerprint.hex()}.json"]
    assert "Failed to cache response" in caplog.text
    assert "injected" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("cookies", [{"session": "value"}, [{"name": "session", "value": "value"}]])
async def test_existing_cache_schema_and_single_writer_replacement(tmp_path, cookies):
    from base64 import b64encode

    import orjson

    cache = ResponseCacheManager(tmp_path)
    fingerprint = b"legacy-cache"
    path = tmp_path / f"{fingerprint.hex()}.json"
    legacy = {
        "url": "https://example.com/legacy",
        "content": b64encode(b"legacy body").decode("ascii"),
        "status": 201,
        "reason": "Created",
        "encoding": "utf-8",
        "cookies": cookies,
        "headers": {"content-type": "text/html"},
        "request_headers": {"user-agent": "legacy"},
        "method": "POST",
    }
    path.write_bytes(orjson.dumps(legacy))
    restored = await cache.get(fingerprint)
    assert restored is not None
    assert restored.body == b"legacy body" and restored.status == 201
    assert restored.url == legacy["url"] and restored.method == "POST"
    assert restored.cookies == (tuple(cookies) if isinstance(cookies, list) else cookies)
    await cache.put(fingerprint, _response(b"new body"), method="PUT")
    replaced = await cache.get(fingerprint)
    assert replaced is not None and replaced.body == b"new body" and replaced.method == "PUT"
    assert set(orjson.loads(path.read_bytes())) == set(legacy)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.anyio
async def test_failed_writer_cleanup_does_not_remove_other_writers_temp(tmp_path, monkeypatch, caplog):
    fingerprint = b"mixed-result"
    first_ready = anyio.Event()
    both_ready = anyio.Event()
    failed_done = anyio.Event()
    paths = []
    original_replace = anyio.Path.replace

    async def controlled_replace(source, target):
        paths.append(str(source))
        if len(paths) == 1:
            first_ready.set()
            await both_ready.wait()
            raise OSError("first writer replacement failed")
        both_ready.set()
        await failed_done.wait()
        return await original_replace(source, target)

    monkeypatch.setattr(anyio.Path, "replace", controlled_replace)

    async def failed_writer():
        await ResponseCacheManager(tmp_path).put(fingerprint, _response(b"failed"))
        failed_done.set()

    async def successful_writer():
        await ResponseCacheManager(tmp_path).put(fingerprint, _response(b"successful"))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(failed_writer)
            await first_ready.wait()
            tasks.start_soon(successful_writer)

    assert len(set(paths)) == 2
    restored = await ResponseCacheManager(tmp_path).get(fingerprint)
    assert restored is not None and restored.body == b"successful"
    assert caplog.text.count("Failed to cache response") == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"{fingerprint.hex()}.json"]
