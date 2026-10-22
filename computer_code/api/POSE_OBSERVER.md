# Read-only PoseStream Observer (independent local host)

This feature is **separate from** the legacy Flask `index.py`. Importing the
legacy module opens a serial device and installs drone arming/setpoint events
at import time. The observer **never imports or modifies that module**.
No active camera data, flight controller, serial commands, pilot controls
or "verified hardware synchronization" status are enabled by this host.

## Opt-in local preview

In one shell:

```sh
cd computer_code/api
python -m pip install numpy scipy Flask Flask-SocketIO
export MOCAP_OBSERVER_VIEWER_TOKEN="REPLACE_WITH_A_LONG_RANDOM_LOCAL_ONLY_SECRET"
python pose_observer_server.py --pattern board-patterns.continuous.example.json --port 3102
```

The server binds **127.0.0.1 only** with the `/pose-observe` Socket.IO
namespace, a required 24–256-character shared viewer token, explicit local
Vite origin allowlist, a fixed subscriber limit, bounded per-viewer frame
queues, and throttled polling. Wildcard CORS and anonymous viewers are
rejected. Use only in a trusted development environment: HTTP on loopback
is **not a production TLS/authentication solution**. The token must not be
placed in the repository, browser bundle, URL query string or logs.

In a second shell (existing Vite project):

```sh
cd computer_code
corepack enable
yarn install --frozen-lockfile
yarn dev --host 127.0.0.1
```

Open [`http://localhost:5173/pose-monitor.html`](http://localhost:5173/pose-monitor.html),
enter the local viewer token and press **Connect as viewer**. The standalone
read-only viewer is compiled separately from the existing `App.tsx`, which
emits flight-control commands on its normal page. Do not use the flight
control page to access the observer.

**Important:** at startup, this isolated observer only knows the
static AssetDescription. It will not invent a camera pose or generate
synthetic "MEASURED" data for display. The monitor remains empty until
a separately authorized, trusted *host-side* integration calls
`PoseObserverHost.publish_frame()` with a validated `LockedBoardFrame`.
The Socket.IO client can **never** publish a pose itself.

## Host-side publisher API

```python
from board_patterns import BoardPatternRegistry
from pose_observer_host import PoseObserverHost

registry = BoardPatternRegistry.from_json_file(
    "board-patterns.continuous.example.json"
)
host = PoseObserverHost(
    registry, viewer_secret=viewer_secret, max_subscribers=8,
    max_pending_frames=4,
)

# Host-owned trusted software ingestion ONLY; no socket client event:
host.publish_frame(current_validated_locked_board_frame)

# A source restart/changed pattern requires a new asset/session barrier:
host.restart_source_session()
# host.replace_pattern_registry(new_registry)

# The independent web service must be started explicitly by the operator
# on loopback. The app factory does not start a listener or background task.
```

This is a **single-owner** ingress: producer callbacks, subscriber lifecycle
and publisher queues are serialized with a lock. Any pose error or unknown
pattern/calibration revision fails closed before it becomes an outgoing
packet. There is no hidden fallback to last known position.

## Wire contract and lifecycle

The independent namespace accepts only:

- `connect` with Socket.IO `auth.token`: compare explicitly and reject
  unknown, missing or oversized credentials; capacity checked on connect
- `disconnect`: immediately revoke that viewer's subscription and frames
- `pose:poll`: bounded local queue flush with minimum poll interval

Only the **host** may call `publish_frame`, `restart_source_session` or
`replace_pattern_registry`. No `arm-drone`, `set-drone-setpoint`,
camera exposure or other flight-controller events are registered.

The host dispatches `mocap:pose-stream` messages **only** in the dedicated
namespace and to the authenticated Socket.IO SID. It sends an immutable
AssetDescription as the first packet; dynamic PoseFrame packets follow
only when a validated source frame is supplied. Reconnect requires a new
AssetDescription before any new PoseFrame. Pattern changes force a new
session and asset barrier. Per-viewer backlog drops old PoseFrames, never
pending assets. The local emit returning success does not mean the remote
client received the event; readers must monitor session/sequence gaps.

The browser client only displays `MEASURED` for a current source trigger
with age zero, and never displays an old quaternion or position when the
body is `INVALID`. It shows declared timestamps, model/calibration revision,
quarantined camera IDs, dropped packets, current marker count, and rigid
and reprojection residuals. It does not perform actuator control.

Source exposure timestamps are declared **nanoseconds** and can exceed
JavaScript's exact integer precision. A web dashboard is therefore a
diagnostic display, not a timing oracle. Precise nanosecond arithmetic,
checksums and protocol validation should use the Python consumer or a
64-bit-integer-aware client. The display must not claim microsecond-accurate
latency measured from JavaScript-rendered timestamps.

## Acceptance and explicit non-goals

[SW #16](https://github.com/qigao/Low-Cost-Mocap/issues/16) tests
actual Flask-SocketIO test-client access control, per-room targeting,
subscriber capacity, independent consumer validation, asset-first
connection and reconnect, invalid/null pose, revision changes and
poll-rate limits. Frontend CI uses strict TypeScript checking for
`tsconfig.pose-monitor.json` and a separate Vite HTML entry so unrelated
existing legacy `App.tsx` type errors do not silently disable checking.

A physical hardware-backed PoseStream producer, measured send-to-receive
network latency, stronger multi-user auth, TLS reverse proxy, and real
sensor-exposure synchronization remain **not implemented or not verified**.
[HW Issues #2–#7](../../docs/HARDWARE_VALIDATION.md) continue to track
physical timing, calibration, optical LED reliability, motion testing
and flight-control safety. Never enable control commands based on this
local read-only observer.
