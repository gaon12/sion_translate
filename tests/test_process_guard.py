from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest
from sion_translate import process_guard


def test_non_linux_platforms_leave_parent_death_signals_disabled(monkeypatch) -> None:
    monkeypatch.setattr(process_guard, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        process_guard,
        "_set_linux_parent_death_signal",
        lambda _signum: pytest.fail("a non-Linux platform must not call prctl"),
    )

    assert process_guard.arm_linux_parent_death_signal(123) is False


def test_linux_parent_death_guard_checks_the_parent_before_and_after_prctl(
    monkeypatch,
) -> None:
    observed_parents = iter((123, 123))
    armed_signals: list[int] = []
    monkeypatch.setattr(process_guard, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        process_guard,
        "os",
        SimpleNamespace(getppid=lambda: next(observed_parents)),
    )
    monkeypatch.setattr(
        process_guard,
        "_set_linux_parent_death_signal",
        armed_signals.append,
    )

    assert process_guard.arm_linux_parent_death_signal(123) is True
    assert armed_signals == [process_guard._LINUX_SIGKILL]


@pytest.mark.parametrize(
    ("observed_parents", "expected_armed_signals"),
    (
        ((999,), []),
        ((123, 999), [process_guard._LINUX_SIGKILL]),
    ),
)
def test_linux_parent_death_guard_rejects_both_sides_of_the_setup_race(
    monkeypatch,
    observed_parents: tuple[int, ...],
    expected_armed_signals: list[int],
) -> None:
    parent_iterator = iter(observed_parents)
    armed_signals: list[int] = []
    monkeypatch.setattr(process_guard, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        process_guard,
        "os",
        SimpleNamespace(getppid=lambda: next(parent_iterator)),
    )
    monkeypatch.setattr(
        process_guard,
        "_set_linux_parent_death_signal",
        armed_signals.append,
    )

    def reject_parent_change(expected: int, observed: int) -> None:
        raise RuntimeError(f"parent changed from {expected} to {observed}")

    monkeypatch.setattr(process_guard, "_die_after_parent_race", reject_parent_change)

    with pytest.raises(RuntimeError, match="parent changed from 123 to 999"):
        process_guard.arm_linux_parent_death_signal(123)

    assert armed_signals == expected_armed_signals


def test_parent_race_uses_an_uncatchable_linux_signal(monkeypatch) -> None:
    delivered_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        process_guard,
        "os",
        SimpleNamespace(
            getpid=lambda: 456,
            kill=lambda pid, signum: delivered_signals.append((pid, signum)),
        ),
    )

    with pytest.raises(RuntimeError, match="expected 123, observed 999"):
        process_guard._die_after_parent_race(123, 999)

    assert delivered_signals == [(456, process_guard._LINUX_SIGKILL)]


def test_guard_wrapper_restores_the_signal_mask_published_by_easy_run(monkeypatch) -> None:
    restored_masks: list[tuple[int, set[int]]] = []
    environment = {
        process_guard.INHERITED_SIGNAL_MASK_ENVIRONMENT: f"{int(signal.SIGINT)},{int(signal.SIGTERM)}"
    }
    monkeypatch.setattr(process_guard, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(
        process_guard,
        "signal",
        SimpleNamespace(pthread_sigmask=lambda how, mask: restored_masks.append((how, mask))),
    )

    process_guard._restore_inherited_signal_mask()

    assert restored_masks == [
        (
            process_guard._POSIX_SIG_SETMASK,
            {int(signal.SIGINT), int(signal.SIGTERM)},
        )
    ]
    assert process_guard.INHERITED_SIGNAL_MASK_ENVIRONMENT not in environment


@pytest.mark.parametrize("raw_signal_mask", ("invalid", "2,,15", "-1"))
def test_guard_wrapper_rejects_malformed_inherited_signal_masks(
    monkeypatch,
    raw_signal_mask: str,
) -> None:
    environment = {process_guard.INHERITED_SIGNAL_MASK_ENVIRONMENT: raw_signal_mask}
    monkeypatch.setattr(process_guard, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(
        process_guard,
        "signal",
        SimpleNamespace(pthread_sigmask=lambda *_args: pytest.fail("invalid mask was applied")),
    )

    with pytest.raises(RuntimeError, match="comma-separated signal numbers"):
        process_guard._restore_inherited_signal_mask()


def test_guard_main_arms_parent_death_before_unblocking_signals(monkeypatch) -> None:
    events: list[str] = []
    fake_sys = SimpleNamespace(argv=["process_guard"])
    fake_os = SimpleNamespace(
        environ={process_guard.EXPECTED_PARENT_PID_ENVIRONMENT: "123"},
        getpid=lambda: 456,
    )
    monkeypatch.setattr(process_guard, "sys", fake_sys)
    monkeypatch.setattr(process_guard, "os", fake_os)
    monkeypatch.setattr(
        process_guard,
        "_parse_arguments",
        lambda _argv: ("launcher", "example.target", []),
    )
    monkeypatch.setattr(
        process_guard,
        "arm_linux_parent_death_signal",
        lambda _parent_pid: events.append("arm") or True,
    )
    monkeypatch.setattr(
        process_guard,
        "_restore_inherited_signal_mask",
        lambda: events.append("restore"),
    )
    monkeypatch.setattr(
        process_guard,
        "runpy",
        SimpleNamespace(run_module=lambda *_args, **_kwargs: events.append("run")),
    )

    process_guard.main([])

    assert events == ["arm", "restore", "run"]


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (None, "is missing"),
        ("not-a-pid", "positive process ID"),
        ("0", "positive process ID"),
    ),
)
def test_guard_rejects_missing_or_invalid_parent_pids(
    monkeypatch,
    value: str | None,
    message: str,
) -> None:
    if value is None:
        monkeypatch.delenv(process_guard.EXPECTED_PARENT_PID_ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(process_guard.EXPECTED_PARENT_PID_ENVIRONMENT, value)

    with pytest.raises(RuntimeError, match=message):
        process_guard._expected_parent_pid()


def test_guard_argument_separator_is_not_forwarded_to_the_target_module() -> None:
    role, module, arguments = process_guard._parse_arguments(
        ["worker", "example.worker", "--", "--config", "training.yaml"]
    )

    assert role == "worker"
    assert module == "example.worker"
    assert arguments == ["--config", "training.yaml"]


@pytest.mark.parametrize(
    ("role", "initial_environment", "expected_parent_pid"),
    (
        (
            "launcher",
            {process_guard.EXPECTED_PARENT_PID_ENVIRONMENT: "123"},
            123,
        ),
        (
            "worker",
            {process_guard.EXPECTED_PARENT_PID_ENVIRONMENT: "321"},
            321,
        ),
    ),
)
def test_guard_main_arms_the_expected_parent_and_executes_the_requested_module(
    monkeypatch,
    role: str,
    initial_environment: dict[str, str],
    expected_parent_pid: int,
) -> None:
    fake_sys = SimpleNamespace(argv=["process_guard"])
    fake_os = SimpleNamespace(
        environ=initial_environment.copy(),
        getpid=lambda: 456,
    )
    armed_parents: list[int] = []
    observed_run: dict[str, object] = {}

    def run_module(module: str, *, run_name: str, alter_sys: bool) -> None:
        observed_run.update(
            module=module,
            run_name=run_name,
            alter_sys=alter_sys,
            argv=list(fake_sys.argv),
            guardian=fake_os.environ[process_guard.EXPECTED_PARENT_PID_ENVIRONMENT],
        )

    monkeypatch.setattr(process_guard, "sys", fake_sys)
    monkeypatch.setattr(process_guard, "os", fake_os)
    monkeypatch.setattr(
        process_guard,
        "_parse_arguments",
        lambda _argv: (role, "example.target", ["--value", "safe"]),
    )
    monkeypatch.setattr(
        process_guard,
        "arm_linux_parent_death_signal",
        lambda parent_pid: armed_parents.append(parent_pid) or True,
    )
    monkeypatch.setattr(process_guard, "runpy", SimpleNamespace(run_module=run_module))

    process_guard.main([])

    assert armed_parents == [expected_parent_pid]
    assert observed_run == {
        "module": "example.target",
        "run_name": "__main__",
        "alter_sys": True,
        "argv": ["example.target", "--value", "safe"],
        "guardian": "456",
    }
