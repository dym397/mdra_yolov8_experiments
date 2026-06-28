from __future__ import annotations

from pathlib import PurePosixPath

from mdra.utils.path_utils import resolve_path, safe_mkdir, unique_experiment_dir


def test_safe_mkdir_and_resolve_path(tmp_path):
    nested = safe_mkdir(tmp_path / "a" / "b")
    resolved = resolve_path(tmp_path, "a/b")
    assert nested.is_dir()
    assert resolved == nested.resolve()


def test_unique_experiment_dir_never_overwrites(tmp_path):
    first = unique_experiment_dir(tmp_path, "B1-visible")
    marker = first / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    second = unique_experiment_dir(tmp_path, "B1-visible")

    assert first != second
    assert marker.read_text(encoding="utf-8") == "keep"
    assert second.is_dir()


def test_posix_paths_remain_portable():
    path = PurePosixPath("/datasets/M3FD/visible/sample.jpg")
    assert path.is_absolute()
    assert path.parts[-3:] == ("M3FD", "visible", "sample.jpg")

