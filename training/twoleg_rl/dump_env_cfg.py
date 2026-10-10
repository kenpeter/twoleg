"""Dump a resolved mjlab velocity env config to normalized JSON.

Runs under either project's venv and reads the *effective* config object, not
the source text, so mjlab base defaults inherited by both projects are included.

Usage (from the project root, using that project's venv):

    ./.venv/bin/python training/scripts/dump_env_cfg.py twoleg  > .audit/twoleg.json
    ./.venv/bin/python scripts/dump_env_cfg.py microduck     > ../.audit/microduck.json
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import mujoco

# Managers to dump. Keys are the JSON top-level sections.
SECTIONS = (
    "rewards",
    "events",
    "terminations",
    "curriculum",
    "commands",
    "observations",
    "actions",
    "sim",
    "scene",
)


def norm(value: Any, depth: int = 0) -> Any:
    """Recursively normalize a config value into JSON-safe comparable data."""
    if depth > 8:
        return "<deep>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # Keep math.sqrt(0.1) etc. comparable across both dumps; round away
        # last-bit noise so 0.31622776601683794 == 0.31622776601683794.
        return round(value, 12)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [norm(v, depth + 1) for v in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=repr)
        return items
    if isinstance(value, dict):
        return {str(k): norm(v, depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, Path):
        return {"__path__": value.name}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {"__class__": f"{type(value).__module__}.{type(value).__qualname__}"}
        # Fields alone are not enough: a cfg can carry a side attribute that no
        # dataclass field declares (e.g. twoleg sets rel_turn_in_place_envs on
        # mjlab's UniformVelocityCommandCfg, which declares no such field).
        # Union both so the dump does not silently drop knobs.
        names = {f.name for f in dataclasses.fields(value)} | set(vars(value))
        for name in sorted(names):
            if name.startswith("_"):
                continue
            out[name] = norm(getattr(value, name, None), depth + 1)
        return out
    if callable(value):
        mod = getattr(value, "__module__", "?")
        qual = getattr(value, "__qualname__", repr(value))
        return {"__func__": f"{mod}.{qual}"}
    # Named tuples and mjlab config objects that are not dataclasses.
    if hasattr(value, "__dict__"):
        out = {"__class__": f"{type(value).__module__}.{type(value).__qualname__}"}
        for k, v in sorted(vars(value).items(), key=lambda kv: str(kv[0])):
            if k.startswith("_"):
                continue
            out[k] = norm(v, depth + 1)
        return out
    if hasattr(value, "__slots__"):
        out = {"__class__": f"{type(value).__module__}.{type(value).__qualname__}"}
        for k in sorted(value.__slots__):
            out[k] = norm(getattr(value, k, None), depth + 1)
        return out
    return {"__repr__": repr(value)}


def dump_terms(getter) -> Any:
    out = {}
    for name, term in sorted(getter().items()):
        entry = {"__class__": f"{type(term).__module__}.{type(term).__qualname__}"}
        for f in ("func", "weight", "params", "interval_range_s", "time_unit"):
            if hasattr(term, f):
                entry[f] = norm(getattr(term, f))
        extra = {
            k: norm(v)
            for k, v in sorted(vars(term).items())
            if k not in entry and not k.startswith("_")
        }
        entry.update(extra)
        out[str(name)] = entry
    return out


def build(project: str) -> dict:
    if project == "twoleg":
        from twoleg_training.tasks.twoleg_velocity_env_cfg import (
            TwoLegRlCfg,
            make_twoleg_velocity_env_cfg,
        )
        from twoleg_rl.robot import JOINT_NAMES, TWOLEG_XML

        cfg = make_twoleg_velocity_env_cfg()
        runner = TwoLegRlCfg
        declared_joints = list(JOINT_NAMES)
        xml_path = TWOLEG_XML
    elif project == "microduck":
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
            MicroduckRlCfg,
            make_microduck_velocity_env_cfg,
        )

        from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML

        cfg = make_microduck_velocity_env_cfg()
        runner = MicroduckRlCfg
        declared_joints = None
        xml_path = MICRODUCK_WALK_XML
    else:
        raise SystemExit(f"unknown project {project!r}")

    # Real DOF count straight from the compiled MuJoCo spec, not from a tuple.
    entity = cfg.scene.entities["robot"]
    spec = entity.spec_fn()
    spec.compile()
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    robot_extra = {
        "n_xml_joints": len(spec.joints),
        "n_actuators": len(spec.actuators),
        "n_bodies": len(spec.bodies),
        "n_geoms": len(spec.geoms),
        "total_mass_kg": round(float(mj_model.body_mass.sum()), 4),
        "actuator_names": [a.name for a in spec.actuators],
        "declared_joint_names": declared_joints,
    }

    data: dict[str, Any] = {"_project": project, "_robot": robot_extra}

    for section in SECTIONS:
        obj = getattr(cfg, section, None)
        if obj is None:
            data[section] = None
        elif section == "observations":
            data[section] = {
                grp: {
                    "enable_corruption": norm(cfg_.enable_corruption),
                    "concatenate_terms": norm(cfg_.concatenate_terms),
                    "terms": dump_terms(lambda g=cfg_: g.terms),
                }
                for grp, cfg_ in sorted(obj.items())
            }
        elif section == "sim":
            data[section] = {
                k: norm(v) for k, v in sorted(vars(obj).items()) if not k.startswith("_")
            }
        elif section == "scene":
            data[section] = {
                "terrain": norm(getattr(obj, "terrain", None)),
                "sensors": sorted(
                    f"{getattr(s, 'name', '?')}:{type(s).__name__}" for s in (obj.sensors or ())
                ),
                "entities": sorted(str(k) for k in (obj.entities or {})),
            }
        elif hasattr(obj, "terms"):
            data[section] = dump_terms(lambda o=obj: o.terms)
        else:
            data[section] = norm(obj)

    # decimation and episode_length_s live on the env cfg, not under sim.
    data["env"] = {
        "decimation": norm(getattr(cfg, "decimation", None)),
        "episode_length_s": norm(getattr(cfg, "episode_length_s", None)),
    }
    data["_runner"] = norm(runner)
    return data


if __name__ == "__main__":
    # argv[1] = project, argv[2] = optional output path. A path is preferred
    # because mjlab_microduck prints [mdp] patch banners on stdout, which would
    # corrupt a JSON stream.
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    text = json.dumps(build(sys.argv[1]), indent=1, sort_keys=True) + "\n"
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(text)
    else:
        sys.stdout.write(text)