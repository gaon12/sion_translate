from __future__ import annotations

import sys

import pytest

from sion_translate.console import configure_stdio


class FakeStream:
    """A text-stream stand-in that records calls to ``reconfigure()``."""

    def __init__(self, encoding: str, *, fails: bool = False):
        self.encoding = encoding
        self.fails = fails
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        if self.fails:
            raise OSError("reconfigure is unavailable here")
        self.calls.append(kwargs)
        self.encoding = kwargs.get("encoding", self.encoding)


class BareStream:
    """A stream without ``reconfigure()``, as seen in some hosts and test runners."""

    encoding = "cp949"


@pytest.fixture(autouse=True)
def _clear_io_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)


def _install(monkeypatch: pytest.MonkeyPatch, *streams: object) -> None:
    for name, stream in zip(("stdin", "stdout", "stderr"), streams, strict=True):
        monkeypatch.setattr(sys, name, stream)


def test_reopens_non_utf8_streams_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    stdin, stdout, stderr = (FakeStream("cp949") for _ in range(3))
    _install(monkeypatch, stdin, stdout, stderr)

    configure_stdio()

    # Make input loss visible while keeping output reversible to the original text.
    assert stdin.calls == [{"encoding": "utf-8", "errors": "replace"}]
    for stream in (stdout, stderr):
        assert stream.calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]


@pytest.mark.parametrize("encoding", ["utf-8", "UTF-8", "utf8"])
def test_leaves_utf8_streams_untouched(monkeypatch: pytest.MonkeyPatch, encoding: str) -> None:
    streams = [FakeStream(encoding) for _ in range(3)]
    _install(monkeypatch, *streams)

    configure_stdio()

    assert all(stream.calls == [] for stream in streams)


def test_honours_explicit_pythonioencoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONIOENCODING", "cp932")
    streams = [FakeStream("cp932") for _ in range(3)]
    _install(monkeypatch, *streams)

    configure_stdio()

    assert all(stream.calls == [] for stream in streams)


def test_tolerates_streams_that_cannot_be_reconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An encoding convenience must not block the CLI, so failure is tolerated.
    _install(monkeypatch, BareStream(), FakeStream("cp949", fails=True), None)

    configure_stdio()
