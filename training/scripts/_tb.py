import os, sys, traceback
REPO = "/home/kenpeter/work/twoleg/training"
TRA = "/home/kenpeter/work/torch-rl-algorithms"
for p in (TRA, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")
try:
    import scripts._smoke_ppo as s
    s.main() if hasattr(s, "main") else None
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
