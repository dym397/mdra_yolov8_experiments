from __future__ import annotations

from pathlib import Path

import pytest

from mdra.experiments.baselines import load_b_experiment_config


def test_b_config_rejects_unknown_variant(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("variant: unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="variant"):
        load_b_experiment_config(path)


def test_all_b_configs_share_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs" / "experiments").glob("B[1-5]_*.yaml"))
    assert len(paths) >= 6
    configs = [load_b_experiment_config(path) for path in paths if path.name in {
        "B1_visible.yaml", "B2_infrared.yaml", "B3_early_fusion.yaml", "B3_early_fusion_p2.yaml",
        "B4_lcmf.yaml", "B5_lcmf_p2.yaml"
    }]
    assert len(configs) == 6
    assert {config["seed"] for config in configs} == {42}
    assert {config["imgsz"] for config in configs} == {640}
    assert {config["effective_batch"] for config in configs} == {16}
    assert {tuple(config["class_names"]) for config in configs} == {
        ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")
    }
