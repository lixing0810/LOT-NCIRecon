"""YAML configuration loading and phase-specific path resolution."""

import copy
from argparse import Namespace
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    return config


def _flatten_sections(config, names):
    values = {}
    for name in names:
        section = config.get(name, {}) or {}
        if not isinstance(section, dict):
            raise ValueError("Configuration section '{}' must be a mapping.".format(name))
        values.update(section)
    return values


def build_phase_config(config, phase):
    """Return the flat Namespace expected by the reconstruction trainer."""
    if phase not in {"A", "V", "D"}:
        raise ValueError("phase must be one of A, V, or D")

    values = _flatten_sections(config, ("data", "model", "training", "loss"))
    output = config.get("output", {}) or {}
    values["phase"] = phase
    values["train_data_dir"] = values.pop("train_template").format(phase=phase)
    monitor_template = values.pop("monitor_template", None)
    values["monitor_data_dir"] = monitor_template.format(phase=phase) if monitor_template else None
    values["output_root"] = output.get("template", "outputs/reconstruction/{phase}").format(phase=phase)

    for name in ("train_manifest", "monitor_manifest", "resume"):
        value = values.get(name)
        if value:
            values[name] = value.format(phase=phase)
    return Namespace(**values)


def override(config, section, **values):
    """Return a deep copy with non-None CLI overrides applied."""
    updated = copy.deepcopy(config)
    target = updated.setdefault(section, {})
    for key, value in values.items():
        if value is not None:
            target[key] = value
    return updated
