#!/usr/bin/env python3
"""Contract suite over every MJCF model in this directory: each model must parse, compile,
stay finite, be physically sane, and keep its named parts; each combination model must
still embed its sub-models verbatim. main() prints the inventory of checks it ran.

Every model here is fully rigid, nq = nv = nu = 0, so the forward/step check only proves
the pipeline recomputes without producing a non-finite value; there is no state to settle.

Run: uv run --offline --with numpy --with mujoco python robot_item/test_model_suite.py
     --accept       rewrite model_graph.json from the current corpus, then exit 0
     --print-graph  print the derived composition edges, then exit 0
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from typing import NamedTuple

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(HERE, "model_graph.json")

WALK_TAGS = ("body", "geom", "joint", "site", "freejoint")
MESH_SCALE = 0.001
QUAT_TOL = 1e-6
MASS_REL_TOL = 1e-9
N_STEPS = 100
SIDES = ("left", "right")

class Row(NamedTuple):
    depth: int
    tag: str
    attrs: tuple
    name: str | None


class ModelSpec(NamedTuple):
    name: str
    path: str
    root: ET.Element | None
    rows: list


class Match(NamedTuple):
    ok: bool
    names: dict
    src_index: int
    host_index: int
    reason: str
    replaced: bool


class Edge(NamedTuple):
    source: str
    host: str
    root: str | None
    rows: int
    host_rows: int
    names: dict
    replaced: bool


class Corpus(NamedTuple):
    by_name: dict
    errors: dict
    graph: dict
    accepted: list
    baseline_error: str
    infos: list


def _walk(el, depth):
    out = []
    for child in el:
        if child.tag not in WALK_TAGS:
            continue
        out.append(Row(depth, child.tag,
                       tuple(sorted((k, v) for k, v in child.attrib.items() if k != "name")),
                       child.get("name")))
        out.extend(_walk(child, depth + 1))
    return out


def rows_of(spec, rootname):
    if rootname is None:
        return spec.rows
    wb = spec.root.find("worldbody")
    host = next((c for c in wb if c.get("name") == rootname), None)
    return [] if host is None else _walk(host, 0)


def discover():
    """Backups and underscore scratch files are not models, so a newly added combination
    model is picked up without a code change."""
    by_name, errors = {}, {}
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith(".xml") or fn.startswith("_") or fn.endswith(".bak"):
            continue
        path = os.path.join(HERE, fn)
        name = fn[:-4]
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:
            by_name[name] = ModelSpec(name, path, None, [])
            errors[name] = str(exc)
            continue
        by_name[name] = ModelSpec(name, path, root, _walk(root.find("worldbody"), 0))
    return by_name, errors


def match_rows(source, host):
    """Ordered-subsequence match of source rows into host rows, names injective and
    consistent. The rename is not a uniform prefix (legs take an R_ prefix, arms an _R
    suffix, the left side none), so the match runs on structure and never on string
    munging.

    Geom rows must match (depth, tag, attrs) exactly. Body rows may differ in pos and
    quat alone, because a combination model is free to re-place a sub-assembly: the leg
    hangs the thigh at its own height rather than the thigh file's own frame. Such an
    edge is still a composition, and `replaced` records that the bodies were re-placed so
    the report can say so instead of claiming a verbatim copy."""

    def key(row):
        if row.tag == "body":
            return (row.depth, row.tag,
                    tuple((k, v) for k, v in row.attrs if k not in ("pos", "quat")))
        return (row.depth, row.tag, row.attrs)

    def full(row):
        return (row.depth, row.tag, row.attrs)

    names, used, h, replaced = {}, set(), 0, False
    for i, s in enumerate(source):
        j = h
        while j < len(host) and full(host[j]) != full(s) and key(host[j]) != key(s):
            j += 1
        if j == len(host):
            return Match(False, {}, i, min(h, len(host) - 1),
                         "no host row left with this depth/tag/attrs", replaced)
        row = host[j]
        if full(row) != full(s):
            replaced = True
        if s.name is not None:
            if row.name is None:
                return Match(False, {}, i, j, "host row carries no name to map onto", replaced)
            if s.name in names:
                if names[s.name] != row.name:
                    return Match(False, {}, i, j,
                                 f"source name {s.name!r} already maps to {names[s.name]!r}",
                                 replaced)
            elif row.name in used:
                return Match(False, {}, i, j, f"host name {row.name!r} is already taken", replaced)
            else:
                names[s.name] = row.name
                used.add(row.name)
        h = j + 1
    return Match(True, names, len(source), len(host) - 1, "", replaced)


def derive_graph(by_name):
    """One match per ordered (source, host) pair and per host candidate root, so a lost
    edge is a diagnosable failure rather than a silent absence."""
    graph = {}
    for src in by_name.values():
        if src.root is None:
            continue
        for host in by_name.values():
            if host is src or host.root is None:
                continue
            wb = host.root.find("worldbody")
            for rname in [None] + [c.get("name") for c in wb if c.get("name")]:
                graph[(src.name, host.name, rname)] = match_rows(src.rows, rows_of(host, rname))
    return graph


def _sort_key(key):
    return (key[0], key[1], key[2] or "")


def load_accepted():
    try:
        with open(GRAPH_PATH, encoding="utf-8") as fh:
            blob = json.load(fh)
        edges = [Edge(e["source"], e["host"], e["root"], e["rows"], e["host_rows"],
                      e["names"], e.get("replaced", False))
                 for e in blob["edges"]]
    except FileNotFoundError:
        return [], f"{os.path.basename(GRAPH_PATH)} is missing"
    except (ValueError, KeyError, TypeError) as exc:
        return [], f"{os.path.basename(GRAPH_PATH)} is unreadable: {exc}"
    return sorted(edges, key=_sort_key), ""


def write_baseline(graph, by_name):
    edges = []
    for (src, host, root), m in graph.items():
        if not m.ok:
            continue
        edges.append({"source": src, "host": host, "root": root,
                      "rows": len(by_name[src].rows),
                      "host_rows": len(rows_of(by_name[host], root)),
                      "names": m.names, "replaced": m.replaced})
    edges.sort(key=lambda e: (e["source"], e["host"], e["root"] or ""))
    with open(GRAPH_PATH, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"version": 1, "edges": edges}, indent=2, ensure_ascii=False) + "\n")
    return len(edges)


def print_graph(graph, by_name):
    n = 0
    for (src, host, root), m in sorted(graph.items(), key=lambda kv: _sort_key(kv[0])):
        if not m.ok:
            continue
        n += 1
        print(f"{src} -> {host}[{root or 'world'}]  "
              f"{len(by_name[src].rows)} of {len(rows_of(by_name[host], root))} rows, "
              f"{len(m.names)} names"
              + ("  [bodies re-placed]" if m.replaced else ""))
        for s_name, h_name in m.names.items():
            print(f"    {s_name} -> {h_name}")
    print(f"{n} composition edges over {len(by_name)} models")


def twin_pairs(by_name):
    """Every left/right filename pair, found by swapping a standalone left or right
    token, so a newly mirrored part is picked up without a code change."""
    pairs = set()
    for name in by_name:
        tokens = name.split("_")
        for i, tok in enumerate(tokens):
            if tok not in SIDES:
                continue
            other = "right" if tok == "left" else "left"
            partner = "_".join(tokens[:i] + [other] + tokens[i + 1:])
            if partner in by_name:
                pairs.add(tuple(sorted((name, partner))))
    return sorted(pairs)


def test_xml_wellformed(corpus):
    fails = [f"{name}: {corpus.errors[name]}" for name in sorted(corpus.errors)]
    return fails, not fails


def test_mujoco_compiles(corpus):
    fails = []
    for spec in corpus.by_name.values():
        if spec.root is None:
            continue
        try:
            mujoco.MjModel.from_xml_path(spec.path)
        except Exception as exc:
            fails.append(f"{spec.name}: {exc}")
    return fails, not fails


def test_forward_and_step_finite(corpus):
    fails, dof = [], set()
    for spec in corpus.by_name.values():
        if spec.root is None:
            continue
        try:
            model = mujoco.MjModel.from_xml_path(spec.path)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for _ in range(N_STEPS):
                mujoco.mj_step(model, data)
        except Exception as exc:
            fails.append(f"{spec.name}: {exc}")
            continue
        dof.add((model.nq, model.nv, model.nu))
        for field in ("qpos", "qvel", "xpos", "xmat"):
            arr = getattr(data, field)
            if not np.isfinite(arr).all():
                first = tuple(int(i) for i in np.argwhere(~np.isfinite(arr))[0])
                fails.append(f"{spec.name}: {field} is not finite at index {first}")
    print(f"  (nq, nv, nu) over the whole corpus: {sorted(dof)}")
    return fails, not fails


def test_mass_and_inertia_physical(corpus):
    """Massless bodies are legal here: the horns and bearings are visual-only and the
    container bodies hold only children. What must hold is that a body owning collision
    geometry is heavy, and that a heavy body has positive inertia on every axis."""
    fails = []
    for spec in corpus.by_name.values():
        if spec.root is None:
            continue
        try:
            model = mujoco.MjModel.from_xml_path(spec.path)
        except Exception as exc:
            fails.append(f"{spec.name}: {exc}")
            continue
        total = float(model.body_mass.sum())
        if not total > 0:
            fails.append(f"{spec.name}: total mass {total} is not > 0")
        heavy = {b.get("name") for b in spec.root.iter("body")
                 if b.get("name") and any(g.get("class") == "collision"
                                          for g in b.findall("geom"))}
        for i in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            mass = float(model.body_mass[i])
            if not math.isfinite(mass):
                fails.append(f"{spec.name}: body {name} mass {mass} is not finite")
            elif mass < 0:
                fails.append(f"{spec.name}: body {name} mass {mass} is negative")
            elif name in heavy and mass <= 0:
                fails.append(f"{spec.name}: body {name} owns a collision geom but has mass {mass}")
            if mass > 0 and not (model.body_inertia[i] > 0).all():
                fails.append(f"{spec.name}: body {name} mass {mass} but inertia "
                             f"{[float(x) for x in model.body_inertia[i]]}")
    return fails, not fails


def test_quats_normalized(corpus):
    """The authored quat attribute, not m.body_quat: MuJoCo normalises the compiled
    value, so the file is the only place a bad quaternion can survive."""
    fails = []
    for spec in corpus.by_name.values():
        if spec.root is None:
            continue
        for body in spec.root.iter("body"):
            q = body.get("quat")
            if not q:
                continue
            try:
                vals = [float(x) for x in q.split()]
            except ValueError:
                fails.append(f"{spec.name}: body {body.get('name')} quat {q!r} is not numeric")
                continue
            norm = math.sqrt(sum(v * v for v in vals))
            if abs(norm - 1.0) > QUAT_TOL:
                fails.append(f"{spec.name}: body {body.get('name')} quat {q} norm {norm:.9f} "
                             f"off by {abs(norm - 1.0):.2e} > {QUAT_TOL}")
    return fails, not fails


def test_mesh_assets_resolve(corpus):
    """STLs are in millimetres and MJCF is in metres, so each scale component must be
    0.001 in magnitude. A sign flip is a mirrored part, not a unit error; anything else
    is a silent factor of a thousand."""
    fails = []
    for spec in corpus.by_name.values():
        if spec.root is None:
            continue
        for mesh in spec.root.iter("mesh"):
            label = mesh.get("name") or mesh.get("file")
            src = mesh.get("file")
            if not src or not os.path.exists(os.path.join(HERE, src)):
                fails.append(f"{spec.name}: mesh {label} file {src!r} does not resolve "
                             f"under {os.path.basename(HERE)}/")
            raw = mesh.get("scale")
            if raw is None:
                fails.append(f"{spec.name}: mesh {label} declares no scale, so the mm to m "
                             f"rule is never applied")
                continue
            try:
                comps = [float(x) for x in raw.split()]
            except ValueError:
                comps = []
            if len(comps) != 3 or any(abs(c) != MESH_SCALE for c in comps):
                fails.append(f"{spec.name}: mesh {label} scale {raw!r} is not "
                             f"{MESH_SCALE} per axis in magnitude")
    return fails, not fails


def test_names_unique(corpus):
    """Every alignment test addresses bodies and geoms by name, so a duplicate silently
    breaks all of them."""
    fails = []
    for spec in corpus.by_name.values():
        if spec.root is None:
            continue
        for tag in ("body", "geom"):
            counts = Counter(e.get("name") for e in spec.root.iter(tag) if e.get("name"))
            for name, k in sorted(counts.items()):
                if k > 1:
                    fails.append(f"{spec.name}: {k} {tag} elements named {name!r}")
    return fails, not fails


def test_collision_has_visual(corpus):
    """Invisible collision geometry is a defect. The reverse is not: the 金属舵盘 horn and
    the bearings are rendered on purpose and deliberately kept out of collision, so a
    visual-only geom is reported as INFO rather than failed."""

    def key(g):
        return (g.get("mesh"), g.get("pos"), g.get("quat"))

    fails = []
    for spec in corpus.by_name.values():
        if spec.root is None:
            continue
        geoms = list(spec.root.iter("geom"))
        visual = {key(g) for g in geoms if g.get("class") == "visual"}
        collision = {key(g) for g in geoms if g.get("class") == "collision"}
        for g in geoms:
            if g.get("class") == "collision" and key(g) not in visual:
                fails.append(f"{spec.name}: collision geom {g.get('name')} mesh "
                             f"{g.get('mesh')} has no visual geom at the same pos/quat")
        only_visual = [g.get("name") or "<unnamed>" for g in geoms
                       if g.get("class") == "visual" and key(g) not in collision]
        if only_visual:
            corpus.infos.append(f"{spec.name}: {len(only_visual)} visual-only geom(s): "
                        f"{', '.join(sorted(only_visual))}")
    return fails, not fails


def _fmt(row):
    if row is None:
        return "(no such row)"
    return f"{row.tag} name={row.name!r} depth={row.depth} attrs={dict(row.attrs)}"


def test_composition_preserved(corpus):
    """Catches a sub-model edit which was never re-synced into its host: the edge stops
    matching, or matches under a different name map, and the row that diverged is named."""
    if corpus.baseline_error:
        return [f"cannot check the accepted graph: {corpus.baseline_error}"], False
    fails = []
    accepted = {(e.source, e.host, e.root) for e in corpus.accepted}
    for e in corpus.accepted:
        head = f"{e.source} -> {e.host}[{e.root or 'world'}]"
        m = corpus.graph.get((e.source, e.host, e.root))
        if m is None:
            fails.append(f"{head} is no longer a composition: {e.host} has no "
                         f"top-level body named {e.root!r}")
            continue
        if not m.ok:
            src = corpus.by_name[e.source].rows
            host = rows_of(corpus.by_name[e.host], e.root)
            s = src[m.src_index] if 0 <= m.src_index < len(src) else None
            h = host[m.host_index] if 0 <= m.host_index < len(host) else None
            fails.append(f"{head} is no longer a composition: {m.reason}; "
                         f"source row {m.src_index} {_fmt(s)} has no counterpart, "
                         f"host is at row {m.host_index} {_fmt(h)}")
            continue
        for key in sorted(set(m.names) | set(e.names)):
            if m.names.get(key) != e.names.get(key):
                fails.append(f"{head}: name map changed, {key!r} was "
                             f"{e.names.get(key)!r} and is now {m.names.get(key)!r}")
        if m.replaced != e.replaced:
            fails.append(f"{head}: bodies were "
                         f"{'re-placed' if e.replaced else 'copied verbatim'} when accepted "
                         f"and are {'re-placed' if m.replaced else 'copied verbatim'} now")
        counts = (len(corpus.by_name[e.source].rows), len(rows_of(corpus.by_name[e.host], e.root)))
        if counts != (e.rows, e.host_rows):
            corpus.infos.append(f"{head}: row counts changed from {e.rows}/{e.host_rows} to "
                        f"{counts[0]}/{counts[1]}, run with --accept to record")
    for key in sorted(corpus.graph, key=_sort_key):
        if corpus.graph[key].ok and key not in accepted:
            corpus.infos.append(f"new composition detected: {key[0]} -> {key[1]}[{key[2] or 'world'}], "
                        f"run with --accept to record")
    return fails, not fails


def test_composition_mass_equivariant(corpus):
    """The geometry-level twin of the verbatim check: strip the host down to the mapped
    geoms and let MuJoCo recompute density x volume, so renames and reformatting cannot
    hide a mass change. Geoms the source hangs off the world body are left out because
    MuJoCo gives the world body no mass, so they weigh nothing there and everything here.
    """
    if corpus.baseline_error:
        return [f"cannot check the accepted graph: {corpus.baseline_error}"], False
    fails = []
    for e in corpus.accepted:
        try:
            want = float(mujoco.MjModel.from_xml_path(corpus.by_name[e.source].path).body_mass.sum())
            world_level = {r.name for r in corpus.by_name[e.source].rows
                           if r.depth == 0 and r.tag == "geom"}
            keep = {n for n in e.names.values() if n not in world_level}
            spec = mujoco.MjSpec.from_file(corpus.by_name[e.host].path)
            for geom in list(spec.geoms):
                if geom.name not in keep:
                    spec.delete(geom)
            got = float(spec.compile().body_mass.sum())
        except Exception as exc:
            fails.append(f"{e.source} -> {e.host}[{e.root or 'world'}]: {exc}")
            continue
        rel = abs(got - want) / max(abs(want), 1e-30)
        if rel > MASS_REL_TOL:
            fails.append(f"{e.source} -> {e.host}[{e.root or 'world'}]: mapped geom mass "
                         f"{got!r} != source total mass {want!r} (rel {rel:.2e} > {MASS_REL_TOL})")
    return fails, not fails


def test_lr_twin_equivalent(corpus):
    fails = []
    for a, b in twin_pairs(corpus.by_name):
        try:
            ma = mujoco.MjModel.from_xml_path(corpus.by_name[a].path)
            mb = mujoco.MjModel.from_xml_path(corpus.by_name[b].path)
        except Exception as exc:
            fails.append(f"{a} <-> {b}: {exc}")
            continue
        ta, tb = float(ma.body_mass.sum()), float(mb.body_mass.sum())
        rel = abs(ta - tb) / max(abs(ta), abs(tb), 1e-30)
        if rel > MASS_REL_TOL:
            fails.append(f"{a} <-> {b}: total mass {ta!r} != {tb!r} (rel {rel:.2e})")
        for field in ("nbody", "ngeom"):
            if getattr(ma, field) != getattr(mb, field):
                fails.append(f"{a} <-> {b}: {field} {getattr(ma, field)} != "
                             f"{getattr(mb, field)}")
        va = sorted(int(x) for x in ma.mesh_vertnum)
        vb = sorted(int(x) for x in mb.mesh_vertnum)
        if va != vb:
            fails.append(f"{a} <-> {b}: mesh vertex counts differ, {va} != {vb}")
    return fails, not fails


def main(argv):
    flags = set(argv[1:])
    unknown = flags - {"--accept", "--print-graph"}
    if unknown:
        print(f"unknown flag: {', '.join(sorted(unknown))}")
        print("usage: test_model_suite.py [--accept | --print-graph]")
        return 2

    by_name, errors = discover()
    graph = derive_graph(by_name)
    derived = sum(1 for m in graph.values() if m.ok)

    if "--print-graph" in flags:
        print_graph(graph, by_name)
        return 0
    if "--accept" in flags:
        n = write_baseline(graph, by_name)
        print(f"wrote {os.path.basename(GRAPH_PATH)}: {n} edges from {len(by_name)} models")
        return 0

    accepted, baseline_error = load_accepted()
    corpus = Corpus(by_name, errors, graph, accepted, baseline_error, [])

    print(f"models: {len(by_name)}   derived composition edges: {derived}   "
          f"accepted edges: {len(accepted)}   baseline: {os.path.basename(GRAPH_PATH)}")
    if baseline_error:
        print(f"  baseline problem: {baseline_error}")

    tests = [
        ("test_xml_wellformed", test_xml_wellformed),
        ("test_mujoco_compiles", test_mujoco_compiles),
        ("test_forward_and_step_finite", test_forward_and_step_finite),
        ("test_mass_and_inertia_physical", test_mass_and_inertia_physical),
        ("test_quats_normalized", test_quats_normalized),
        ("test_mesh_assets_resolve", test_mesh_assets_resolve),
        ("test_names_unique", test_names_unique),
        ("test_collision_has_visual", test_collision_has_visual),
        ("test_composition_preserved", test_composition_preserved),
        ("test_composition_mass_equivariant", test_composition_mass_equivariant),
        ("test_lr_twin_equivalent", test_lr_twin_equivalent),
    ]
    fails_total = []
    for name, fn in tests:
        print(f"\n== {name} ==")
        fails, ok = fn(corpus)
        for f in fails:
            print("  FAIL:", f)
        if ok:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            fails_total.extend(fails or [name])

    if corpus.infos:
        print("\n== INFO ==")
        for line in corpus.infos:
            print("  INFO:", line)

    print("\n" + "=" * 60)
    if fails_total:
        print(f"VERDICT: FAIL - {len(fails_total)} issues")
        for f in fails_total:
            print("  -", f)
        return 1
    print("VERDICT: PASS - all tests green")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
