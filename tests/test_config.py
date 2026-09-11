from pathlib import Path

from lot_ncirecon.config import build_phase_config, load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_covers_all_three_phases():
    config = load_config(CONFIG)
    assert config["phases"] == ["A", "V", "D"]


def test_phase_paths_are_independent():
    config = load_config(CONFIG)
    arterial = build_phase_config(config, "A")
    delayed = build_phase_config(config, "D")
    assert arterial.train_data_dir.endswith("A")
    assert delayed.train_data_dir.endswith("D")
    assert arterial.output_root != delayed.output_root


def test_paper_spatial_constraint_is_50_pixels():
    config = load_config(CONFIG)
    assert config["selection"]["spatial_constraint"] == 50


def test_two_stage_patch_selection_defaults_are_declared():
    config = load_config(CONFIG)
    assert config["candidate"]["top_k"] == 10000
    assert config["candidate"]["quality_threshold"] == 0.735
    assert config["candidate"]["search_radius"] == 20
    assert config["selection"]["train_size"] == 10000
