"""Start a separate loopback-only, observation-only Socket.IO server.

Requires MOCAP_OBSERVER_VIEWER_TOKEN in the process environment.
No drone serial, PS3Eye, camera polling, armed controls or simulated hardware
data are enabled. Host-side publish_frame() is the explicit trusted ingress.

Example:
  MOCAP_OBSERVER_VIEWER_TOKEN='<random 24+ character viewer secret>' \
    python pose_observer_server.py --pattern board-patterns.continuous.example.json
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from board_patterns import BoardPatternRegistry
from pose_observer_host import PoseObserverHost


def create_local_observer(
    path: str | Path,
    viewer_secret: str,
    *,
    max_subscribers: int = 8,
    max_pending_frames: int = 4,
) -> PoseObserverHost:
    """Factory for an isolated, no-hardware localhost service."""
    return PoseObserverHost(
        BoardPatternRegistry.from_json_file(path),
        viewer_secret=viewer_secret,
        max_subscribers=max_subscribers,
        max_pending_frames=max_pending_frames,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern", type=Path,
        default=Path(__file__).with_name(
            "board-patterns.continuous.example.json"
        ),
    )
    parser.add_argument("--port", type=int, default=3102)
    args = parser.parse_args(argv)
    if type(args.port) is not int or not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    secret = os.environ.get("MOCAP_OBSERVER_VIEWER_TOKEN")
    if not secret:
        parser.error("MOCAP_OBSERVER_VIEWER_TOKEN is required; no anonymous mode")
    try:
        server = create_local_observer(args.pattern, secret)
    except (ValueError, OSError) as exc:
        parser.error(f"invalid observer configuration: {exc}")
    # Deliberately no host argument: it binds loopback only. Development
    # Werkzeug mode is never exposed on 0.0.0.0 by this executable.
    server.socketio.run(
        server.app,
        host="127.0.0.1", port=args.port,
        debug=False, use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
