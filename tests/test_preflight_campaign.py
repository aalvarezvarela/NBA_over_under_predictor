from pathlib import Path

from scripts.preflight_campaign import _resolve_config_paths


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("experiment_name: test\n")
    return path


def test_resolve_config_paths_accepts_an_explicit_subset(tmp_path):
    first = _touch(tmp_path / "first.yaml")
    second = _touch(tmp_path / "second.yml")
    _touch(tmp_path / "not_selected.yaml")

    configs, errors = _resolve_config_paths([first, second, first])

    assert errors == []
    assert configs == [first, second]


def test_resolve_config_paths_expands_directories_and_ignores_base(tmp_path):
    first = _touch(tmp_path / "first.yaml")
    second = _touch(tmp_path / "second.yml")
    _touch(tmp_path / "_base.yaml")
    _touch(tmp_path / "notes.txt")

    configs, errors = _resolve_config_paths([tmp_path])

    assert errors == []
    assert configs == [first, second]
