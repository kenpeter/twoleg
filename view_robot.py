#!/usr/bin/env python3
"""View the full robot XML (interactive viewer or headless PNG).

Usage:
  training/.venv/bin/python view_robot.py [--xml twoleg_mjcf/robot_twoleg.xml] [--out render.png] [--headless]
"""
import argparse, os, sys

DEFAULT_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "twoleg_mjcf", "robot_twoleg.xml")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--out", default="render_full_robot.png")
    ap.add_argument("--headless", action="store_true",
                    help="offscreen render to --out instead of interactive viewer")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="simulate N seconds in headless mode (0 = single frame)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl" if args.headless or not os.environ.get("DISPLAY") else "glfw")
    import mujoco

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"loaded: {args.xml}  nbodies={model.nbody} njoints={model.njnt} ngeom={model.ngeom}")

    if args.headless or not os.environ.get("DISPLAY"):
        import numpy as np
        steps = int(args.seconds / model.opt.timestep) if args.seconds > 0 else 0
        for _ in range(steps):
            mujoco.mj_step(model, data)
        renderer = mujoco.Renderer(model, 1024, 1024)
        renderer.update_scene(data)
        img = renderer.render()
        try:
            import imageio.v2 as imageio
            imageio.imwrite(args.out, img)
        except ImportError:
            import numpy as _np  # fallback via mediapy
            import mediapy as media
            media.write_image(args.out, _np.asarray(img))
        print(f"wrote {args.out}")
    else:
        import mujoco.viewer
        mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
