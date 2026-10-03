"""Local web dashboard for a live TwoLeg training run.

Stdlib only, reads TensorBoard scalars read-only, binds 127.0.0.1.

    uv run python scripts/watch_gait_ui.py
    uv run python scripts/watch_gait_ui.py --port 9000 --run 2026-09-28_21-56-57_velocity

Shows the terms that decide whether the policy walks, against targets taken from
microduck_rl's own velocity run, plus a sparkline per term and the newest
rendered training video.
"""

import argparse
import glob
import html
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from tensorboard.backend.event_processing import event_accumulator

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(TRAINING_DIR, "logs", "rsl_rl", "velocity")

# (tag, comparator, target, why it matters)
TILES = (
    ("Episode_Reward/track_linear_velocity", "ge", 1.0, "tracks the velocity command"),
    ("Episode_Reward/air_time", "ge", 0.5, "the only reward that pays for a swing phase"),
    ("Metrics/air_time_mean", "ge", 0.10, "seconds a foot stays off the ground"),
    ("Metrics/twist/error_vel_xy", "lt", 0.40, "velocity tracking error, m/s"),
    ("Episode_Reward/upright", "ge", 1.5, "torso stays vertical"),
    ("Episode_Reward/pose", "ge", 0.5, "posture, looser once walking"),
    ("Metrics/slip_velocity_mean", "lt", 0.05, "foot dragging on the floor"),
    ("Episode_Termination/fell_over", "lt", 5.0, "percent of episodes ending in a fall"),
    ("Episode_Termination/nan_state", "eq", 0.0, "divergent states caught by the guard"),
    ("Train/mean_episode_length", "gt", 800.0, "out of a 1000 step cap"),
)

SPARK = (
    "Episode_Reward/air_time",
    "Metrics/air_time_mean",
    "Metrics/twist/error_vel_xy",
    "Train/mean_reward",
)

_lock = threading.Lock()
_cache = {"run": None, "payload": None, "at": 0.0}


def runs():
    found = [
        d
        for d in glob.glob(os.path.join(LOG_ROOT, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "events.out.tfevents.*"))
    ]
    return sorted(found, key=os.path.getmtime, reverse=True)


def event_file(run):
    return glob.glob(os.path.join(run, "events.out.tfevents.*"))[0]


def read_run(run):
    ea = event_accumulator.EventAccumulator(event_file(run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    step = 0
    series = {}
    for tag, *_ in TILES + tuple((t,) for t in SPARK):
        if tag not in tags:
            series[tag] = []
            continue
        s = ea.Scalars(tag)
        series[tag] = [(x.step, x.value) for x in s]
        if s:
            step = max(step, s[-1].step)
    return step, series, sorted(tags)


def window(points, frac=0.05, cap=100):
    if not points:
        return None
    n = max(1, int(len(points) * frac))
    return sum(v for _, v in points[-n:]) / min(n, cap)


def check(value, comp, target):
    if value is None:
        return "unknown"
    ok = {
        "ge": value >= target,
        "gt": value > target,
        "lt": value < target,
        "le": value <= target,
        "eq": abs(value - target) < 1e-9,
    }[comp]
    return "ok" if ok else "fail"


def sparkline(points, width=220, height=34, invert=False):
    if len(points) < 2:
        return '<svg class="spark" width="%d" height="%d"></svg>' % (width, height)
    vals = [v for _, v in points]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    pts = " ".join(
        "%.1f,%.1f"
        % (
            2 + i * (width - 4) / (n - 1),
            height - 3 - (height - 6) * (1 - (v - lo) / span) if not invert
            else 3 + (height - 6) * (v - lo) / span,
        )
        for i, v in enumerate(vals)
    )
    return (
        f'<svg class="spark" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><polyline points="{pts}"/></svg>'
    )


def build(run):
    step, series, tags = read_run(run)
    cards = []
    for tag, comp, target, why in TILES:
        if tag not in tags:
            cards.append(
                f'<div class="card unknown"><div class="t">{html.escape(tag.split("/")[-1])}'
                f"</div><div class=v>absent</div><div class=why>not in this run</div></div>"
            )
            continue
        val = window(series[tag])
        state = check(val, comp, target)
        op = {"ge": "≥", "gt": ">", "lt": "<", "le": "≤", "eq": "="}[comp]
        cards.append(
            f'<div class="card {state}"><div class="t">{html.escape(tag.split("/")[-1])}</div>'
            f'<div class="v">{val:.4g}</div>'
            f'<div class="why">target {op} {target} &middot; {html.escape(why)}</div></div>'
        )
    sparks = "".join(
        f'<div class="row"><span class="lbl">{html.escape(t.split("/")[-1])}</span>'
        f'{sparkline(series[t], invert=(t == "Metrics/twist/error_vel_xy"))}</div>'
        for t in SPARK
        if series.get(t)
    )
    videos = sorted(
        glob.glob(os.path.join(run, "videos", "**", "*.mp4"), recursive=True),
        key=os.path.getmtime,
    )
    vid = (
        f'<a class="vid" href="/video">latest training video ({len(videos)} so far)</a>'
        if videos
        else '<span class="novid">no video rendered yet</span>'
    )
    passed = sum(1 for c in cards if "card ok" in c)
    return f"""<!doctype html><meta charset=utf-8>
<meta http-equiv=refresh content=15>
<style>
 body{{background:#0d1117;color:#c9d1d9;font:14px/1.5 ui-monospace,Menlo,monospace;margin:0;padding:24px}}
 h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#8b949e;margin-bottom:18px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
 .card{{background:#161b22;border:1px solid #30363d;border-left-width:4px;border-radius:6px;padding:12px}}
 .card.ok{{border-left-color:#3fb950}} .card.fail{{border-left-color:#f85149}}
 .card.unknown{{border-left-color:#8b949e;opacity:.55}}
 .t{{color:#8b949e;font-size:11px;letter-spacing:.03em}}
 .v{{font-size:26px;margin:4px 0}} .fail .v{{color:#f85149}} .ok .v{{color:#3fb950}}
 .why{{color:#6e7681;font-size:11px}}
 .row{{display:flex;align-items:center;gap:12px;margin-top:6px}}
 .lbl{{width:170px;color:#8b949e;font-size:12px;text-align:right}}
 .spark polyline{{fill:none;stroke:#58a6ff;stroke-width:1.5}}
 h2{{font-size:12px;color:#8b949e;margin:22px 0 6px;font-weight:600;letter-spacing:.06em}}
 a.vid{{color:#58a6ff}} .novid{{color:#6e7681}}
 .bar{{height:6px;background:#21262d;border-radius:3px;margin:10px 0 20px;overflow:hidden}}
 .bar i{{display:block;height:100%;background:#3fb950}}
</style>
<h1>TwoLeg gait watch</h1>
<div class=sub>{html.escape(os.path.basename(run.rstrip('/')))} &nbsp; iteration <b>{step}</b>
 &nbsp; refreshed {time.strftime('%H:%M:%S')} &nbsp; auto-reloads every 15s</div>
<div class=bar><i style="width:{100*passed//max(1,len(cards))}%"></i></div>
<div class=sub>{passed} of {len(cards)} targets met</div>
<div class=grid>{''.join(cards)}</div>
<h2>TRENDS (lower is better for the velocity error)</h2>
{sparks}
<h2>VIDEO</h2>
{vid}
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _send(self, body, ctype):
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        requested = self.path.lstrip("/").split("?")[0]
        available = runs()
        if not available:
            return self._send("<h1>no runs found</h1>", "text/html")
        run = os.path.join(LOG_ROOT, requested) if requested else available[0]
        if not os.path.isdir(run):
            run = available[0]
        if self.path.startswith("/video"):
            vids = sorted(
                glob.glob(os.path.join(run, "videos", "**", "*.mp4"), recursive=True),
                key=os.path.getmtime,
            )
            if not vids:
                return self._send("no video yet", "text/plain")
            with open(vids[-1], "rb") as fh:
                return self._send(fh.read(), "video/mp4")
        with _lock:
            now = time.time()
            if _cache["run"] != run or not _cache["payload"] or now - _cache["at"] > 5:
                _cache.update(run=run, payload=build(run), at=now)
            page = _cache["payload"]
        nav = "".join(
            f'<option{" selected" if os.path.basename(d.rstrip("/")) == requested else ""}>'
            f'{os.path.basename(d.rstrip("/"))}</option>'
            for d in available[:12]
        )
        page = page.replace(
            "<h2>VIDEO</h2>",
            f'<h2>RUN</h2><select onchange="location.href=\'/?\'+this.value">'
            f"{nav}</select><h2>VIDEO</h2>",
        )
        self._send(page, "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--run", default="")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/" + (args.run + "?" if args.run else "")
    print(f"gait watch on {url}   (ctrl-c to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
