from pathlib import Path


def test_run_dev_quiets_uvicorn_info_and_access_logs() -> None:
    script = Path("scripts/run_dev.ps1").read_text(encoding="utf-8")

    assert "--log-level warning" in script
    assert "--no-access-log" in script


def test_run_dev_uses_project_python_module_entrypoint() -> None:
    script = Path("scripts/run_dev.ps1").read_text(encoding="utf-8")

    assert "& $PythonBin -m uvicorn app:app" in script
    assert ".venv/bin/uvicorn" not in script
