import { io, type Socket } from 'socket.io-client'
import './pose-monitor.css'

// This standalone entry deliberately DOES NOT import App.tsx, which emits
// arm-drone and set-drone-setpoint events from its React lifecycle.
const NS = '/pose-observe'
type Obj = Record<string, unknown>
type Model = {
  board_id: string
  type: string
  marker_ids: number[]
  min_visible_markers: number
}

function isObject(value: unknown): value is Obj {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
function isNonnegativeInt(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}
// Real nanoseconds since an epoch may exceed JavaScript's safe integer
// precision. Accept finite display-only timestamps; never claim exact
// timing or perform sub-microsecond arithmetic in the browser.
function isDisplayTimestamp(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0
}
function finiteVector(value: unknown, count: number): value is number[] {
  return Array.isArray(value) && value.length === count
    && value.every(x => typeof x === 'number' && Number.isFinite(x))
}
function unitQuaternion(value: unknown): value is number[] {
  if (!finiteVector(value, 4)) return false
  const norm = value.reduce((sum, x) => sum + x * x, 0)
  return Math.abs(norm - 1) <= 1e-5
}
function isSHA256(value: unknown): value is string {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value)
}
function get<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id)
  if (found === null) throw new Error('Missing observer element: ' + id)
  return found as T
}
function text(value: unknown): string {
  if (typeof value === 'string' && value.length < 150) return value
  if (isNonnegativeInt(value)) return String(value)
  return '—'
}
function vec(value: unknown, count: number): string {
  if (!Array.isArray(value) || value.length !== count) return '—'
  if (!value.every(x => typeof x === 'number' && Number.isFinite(x))) return '—'
  return value.map(x => (x as number).toFixed(4)).join(', ')
}
const serverUrl = get<HTMLInputElement>('server-url')
const token = get<HTMLInputElement>('viewer-token')
const startButton = get<HTMLButtonElement>('connect-button')
const stopButton = get<HTMLButtonElement>('disconnect-button')
const indicator = get<HTMLElement>('connection')
const message = get<HTMLElement>('message')
const rows = get<HTMLTableSectionElement>('rigid-bodies')
const frameElement = get<HTMLElement>('frame-trigger')
const exposureElement = get<HTMLElement>('exposure-time')
const revisionElement = get<HTMLElement>('pattern-revision')
const dropsElement = get<HTMLElement>('drop-count')
const boardSummary = get<HTMLElement>('board-summary')
const patternSHA = get<HTMLElement>('config-sha')
const calibrationSHA = get<HTMLElement>('calibration-sha')
const quarantined = get<HTMLElement>('quarantined')
const clockDomain = get<HTMLElement>('clock-domain')

let socket: Socket | null = null
let pollTimer: number | null = null
let session: number | null = null
let sequence: number | null = null
let frame: number | null = null
let revision: string | null = null
let models: Model[] = []
let gaps = 0
let dropped = 0
let lastFrameReceivedAt: number | null = null
let displayedBodies: Obj[] = []
let staleShown = false
const DISPLAY_SILENCE_LIMIT_MS = 1000

function clearCurrent(reason: string): void {
  session = null
  sequence = null
  frame = null
  revision = null
  models = []
  gaps = 0
  dropped = 0
  lastFrameReceivedAt = null
  displayedBodies = []
  staleShown = false
  frameElement.textContent = '—'
  exposureElement.textContent = '—'
  revisionElement.textContent = '—'
  dropsElement.textContent = '0 / 0'
  boardSummary.textContent = reason
  patternSHA.textContent = '—'
  calibrationSHA.textContent = '—'
  quarantined.textContent = '—'
  clockDomain.textContent = '—'
  rows.replaceChildren()
  const blank = document.createElement('tr')
  const cell = document.createElement('td')
  cell.colSpan = 7
  cell.className = 'empty'
  cell.textContent = 'No current measured PoseFrame.'
  blank.appendChild(cell)
  rows.appendChild(blank)
}

function drawBodies(bodies: Obj[], trigger: number, stale = false): void {
  rows.replaceChildren()
  for (const model of models) {
    const body = bodies.find(b => b.board_id === model.board_id)
    const valid = body !== undefined
      && !stale
      && body.tracking_status === 'TRACKING'
      && body.evidence_kind === 'MEASURED'
      && body.measurement_trigger_id === trigger
      && body.measurement_age_triggers === 0
      && body.measurement_age_ns === 0
      && finiteVector(body.position_world_from_body, 3)
      && unitQuaternion(body.quaternion_world_from_body_xyzw)
      && finiteVector(body.position_world_from_board, 3)
      && unitQuaternion(body.quaternion_world_from_board_xyzw)
      && isObject(body.quality)
      && typeof body.quality.rigid_rms_world_units === 'number'
      && Number.isFinite(body.quality.rigid_rms_world_units)
      && body.quality.rigid_rms_world_units >= 0
      && typeof body.quality.reprojection_max_rms_px === 'number'
      && Number.isFinite(body.quality.reprojection_max_rms_px)
      && body.quality.reprojection_max_rms_px >= 0
    const visible = body !== undefined && Array.isArray(body.visible_marker_ids)
      ? body.visible_marker_ids.filter(isNonnegativeInt).join(', ') : '—'
    const quality = valid && isObject(body.quality) ? body.quality : null
    const content = [
      model.board_id,
      stale ? 'STALE · no recent source frame'
        : valid ? 'MEASURED' : text(body?.evidence_kind ?? 'INVALID'),
      visible || '—',
      valid ? vec(body.position_world_from_body, 3) : '—',
      valid ? vec(body.quaternion_world_from_body_xyzw, 4) : '—',
      stale ? 'STALE' : text(body?.measurement_age_ns),
      quality === null ? '—'
        : text(quality.rigid_rms_world_units) + ' / ' + text(quality.reprojection_max_rms_px),
    ]
    const row = document.createElement('tr')
    content.forEach((value, index) => {
      const col = document.createElement(index === 0 ? 'th' : 'td')
      col.textContent = value
      if (index === 1) col.className = valid ? 'quality-good' : 'quality-bad'
      row.appendChild(col)
    })
    rows.appendChild(row)
  }
}

function hidePoseIfReceiveStreamIsStale(): void {
  if (lastFrameReceivedAt === null || staleShown) return
  if (performance.now() - lastFrameReceivedAt <= DISPLAY_SILENCE_LIMIT_MS) return
  staleShown = true
  drawBodies(displayedBodies, -1, true)
  boardSummary.textContent = 'No new PoseFrame for over 1 second · cached poses hidden'
  message.textContent = 'STALE stream: last positions are hidden until a new source frame arrives.'
}

function acceptEnvelope(raw: unknown): void {
  if (!isObject(raw) || raw.protocol !== 'low_cost_mocap_stream'
      || raw.schema_version !== 1 || !isNonnegativeInt(raw.session_id)
      || !isNonnegativeInt(raw.sequence) || !isObject(raw.payload)) {
    message.textContent = 'Rejected malformed PoseStream envelope.'
    return
  }
  const payload = raw.payload
  if (payload.protocol !== 'low_cost_mocap_pose' || payload.schema_version !== 1
      || payload.hardware_provenance_verified !== false) {
    message.textContent = 'Rejected unqualified or incompatible pose payload.'
    return
  }
  if (payload.kind === 'asset_description') {
    if (raw.sequence !== 0 || (session !== null && raw.session_id <= session)
        || !isSHA256(payload.pattern_sha256)
        || !isNonnegativeInt(payload.period_triggers) || payload.period_triggers < 4
        || !isNonnegativeInt(payload.max_bit_errors)
        || payload.max_bit_errors >= payload.period_triggers
        || !Array.isArray(payload.boards) || payload.boards.length === 0
        || payload.model_coordinates_are_declared_not_metrologically_verified !== true) {
      message.textContent = 'Rejected stale or malformed asset description.'
      return
    }
    const candidates: Model[] = []
    const allMarkerIds = new Set<number>()
    for (const board of payload.boards) {
      if (!isObject(board) || typeof board.board_id !== 'string'
          || board.board_id.length === 0 || board.board_id.length > 64
          || typeof board.type !== 'string'
          || !isNonnegativeInt(board.min_visible_markers)
          || !Array.isArray(board.markers) || board.markers.length < 3
          || board.min_visible_markers < 3
          || board.min_visible_markers > board.markers.length) {
        message.textContent = 'Rejected malformed board identities or marker minimum.'
        return
      }
      const markerIds: number[] = []
      for (const marker of board.markers) {
        if (!isObject(marker) || !isNonnegativeInt(marker.marker_id)
            || allMarkerIds.has(marker.marker_id)
            || typeof marker.code !== 'string'
            || marker.code.length !== payload.period_triggers
            || !/^[01]+$/.test(marker.code)
            || !finiteVector(marker.xyz_m, 3)) {
          message.textContent = 'Rejected duplicate or malformed static marker definition.'
          return
        }
        allMarkerIds.add(marker.marker_id)
        markerIds.push(marker.marker_id)
      }
      candidates.push({
        board_id: board.board_id,
        type: board.type,
        marker_ids: markerIds,
        min_visible_markers: board.min_visible_markers,
      })
    }
    if (new Set(candidates.map(b => b.board_id)).size !== candidates.length) {
      message.textContent = 'Rejected duplicate board identities.'
      return
    }
    session = raw.session_id
    sequence = raw.sequence
    frame = null
    lastFrameReceivedAt = null
    displayedBodies = []
    staleShown = false
    gaps = 0
    dropped = 0
    revision = payload.pattern_sha256
    models = candidates
    patternSHA.textContent = revision
    revisionElement.textContent = revision.slice(0,12)
    boardSummary.textContent = String(models.length) + ' boards registered · awaiting measurements'
    message.textContent = 'Assets accepted; no current pose yet.'
    drawBodies([], -1)
    return
  }
  const cameraIds = payload.camera_ids
  const quarantinedIds = payload.quarantined_camera_ids
  if (payload.kind !== 'pose_frame' || session !== raw.session_id
      || revision === null || payload.pattern_sha256 !== revision
      || !isNonnegativeInt(payload.frame_id)
      || !isDisplayTimestamp(payload.latest_exposure_timestamp_ns)
      || !isSHA256(payload.calibration_sha256)
      || typeof payload.clock_domain !== 'string' || payload.clock_domain.length === 0
      || !Array.isArray(cameraIds) || cameraIds.length === 0
      || !cameraIds.every(isNonnegativeInt)
      || new Set(cameraIds).size !== cameraIds.length
      || !Array.isArray(quarantinedIds)
      || !quarantinedIds.every(isNonnegativeInt)
      || new Set(quarantinedIds).size !== quarantinedIds.length
      || !quarantinedIds.every(id => cameraIds.includes(id))
      || !isNonnegativeInt(payload.exposure_skew_ns)
      || payload.hardware_provenance_verified !== false
      || payload.position_units !== 'uncalibrated_world'
      || !Array.isArray(payload.rigid_bodies)
      || payload.rigid_bodies.length !== models.length) {
    message.textContent = 'Rejected incompatible or stale PoseFrame.'
    return
  }
  if ((sequence !== null && raw.sequence <= sequence)
      || (frame !== null && payload.frame_id <= frame)) {
    message.textContent = 'Rejected out-of-order source or wire sequence.'
    return
  }
  const bodies: Obj[] = []
  for (const body of payload.rigid_bodies) {
    if (!isObject(body) || typeof body.board_id !== 'string'
        || !models.some(model => model.board_id === body.board_id)
        || bodies.some(other => other.board_id === body.board_id)) {
      message.textContent = 'Rejected missing or unknown board.'
      return
    }
    const model = models.find(item => item.board_id === body.board_id)
    const visibleMarkerIds = body.visible_marker_ids
    const excludedMarkerIds = body.excluded_marker_ids
    if (model === undefined || !Array.isArray(visibleMarkerIds)
        || !Array.isArray(excludedMarkerIds)
        || !visibleMarkerIds.every(isNonnegativeInt)
        || !excludedMarkerIds.every(isNonnegativeInt)
        || new Set(visibleMarkerIds).size !== visibleMarkerIds.length
        || new Set(excludedMarkerIds).size !== excludedMarkerIds.length
        || visibleMarkerIds.some(id => excludedMarkerIds.includes(id))
        || [...visibleMarkerIds, ...excludedMarkerIds]
          .some(id => !model.marker_ids.includes(id))
        || body.metric_scale_validated !== false) {
      message.textContent = 'Rejected marker ownership or scale metadata.'
      return
    }
    const poseFields = [
      'position_world_from_body', 'quaternion_world_from_body_xyzw',
      'position_world_from_board', 'quaternion_world_from_board_xyzw',
    ]
    const measured = body.evidence_kind === 'MEASURED'
    if (measured) {
      if (body.tracking_status !== 'TRACKING'
          || body.measurement_trigger_id !== payload.frame_id
          || body.measurement_age_triggers !== 0
          || body.last_measurement_exposure_timestamp_ns !== payload.latest_exposure_timestamp_ns
          || body.measurement_age_ns !== 0
          || visibleMarkerIds.length < model.min_visible_markers
          || !finiteVector(body.position_world_from_body, 3)
          || !unitQuaternion(body.quaternion_world_from_body_xyzw)
          || !finiteVector(body.position_world_from_board, 3)
          || !unitQuaternion(body.quaternion_world_from_board_xyzw)
          || !isObject(body.quality)
          || typeof body.quality.rigid_rms_world_units !== 'number'
          || !Number.isFinite(body.quality.rigid_rms_world_units)
          || body.quality.rigid_rms_world_units < 0
          || typeof body.quality.reprojection_max_rms_px !== 'number'
          || !Number.isFinite(body.quality.reprojection_max_rms_px)
          || body.quality.reprojection_max_rms_px < 0) {
        message.textContent = 'Rejected stale or malformed MEASURED body.'
        return
      }
    } else if (body.evidence_kind !== 'INVALID'
        || !['DISCOVER', 'DEGRADED', 'LOST'].includes(String(body.tracking_status))
        || poseFields.some(key => body[key] !== null)
        || body.quality !== null) {
      message.textContent = 'Rejected invalid body with cached pose or unknown status.'
      return
    }
    const sourceTrigger = body.measurement_trigger_id
    const triggerAge = body.measurement_age_triggers
    if ((sourceTrigger === null) !== (triggerAge === null)
        || (sourceTrigger !== null && (
          !isNonnegativeInt(sourceTrigger) || !isNonnegativeInt(triggerAge)
          || sourceTrigger > payload.frame_id
          || payload.frame_id - sourceTrigger !== triggerAge
        ))) {
      message.textContent = 'Rejected inconsistent measurement trigger age.'
      return
    }
    const sourceTime = body.last_measurement_exposure_timestamp_ns
    const ageNs = body.measurement_age_ns
    if ((sourceTime === null) !== (ageNs === null)
        || (sourceTime !== null && (
          !isDisplayTimestamp(sourceTime) || !isDisplayTimestamp(ageNs)
          || sourceTime > payload.latest_exposure_timestamp_ns
          || (Number.isSafeInteger(sourceTime)
            && Number.isSafeInteger(payload.latest_exposure_timestamp_ns)
            && payload.latest_exposure_timestamp_ns - sourceTime !== ageNs)
        ))) {
      message.textContent = 'Rejected inconsistent measurement exposure age.'
      return
    }
    bodies.push(body)
  }
  if (sequence !== null) gaps += raw.sequence - sequence - 1
  sequence = raw.sequence
  frame = payload.frame_id
  displayedBodies = bodies
  lastFrameReceivedAt = performance.now()
  staleShown = false
  frameElement.textContent = String(frame)
  exposureElement.textContent = (
    payload.latest_exposure_timestamp_ns > Number.MAX_SAFE_INTEGER ? '≈ ' : ''
  ) + String(payload.latest_exposure_timestamp_ns)
  calibrationSHA.textContent = text(payload.calibration_sha256)
  clockDomain.textContent = text(payload.clock_domain)
  quarantined.textContent = Array.isArray(payload.quarantined_camera_ids)
    ? (payload.quarantined_camera_ids.filter(isNonnegativeInt).join(', ') || 'None') : 'Unknown'
  dropsElement.textContent = String(dropped) + ' / ' + String(gaps)
  drawBodies(bodies, frame)
  boardSummary.textContent = String(
    bodies.filter(body => body.evidence_kind === 'MEASURED').length
  ) + '/' + String(models.length) + ' reported MEASURED at trigger ' + String(frame)
  message.textContent = 'Software PoseFrame received; hardware not verified.'
}

function connectionState(connected: boolean, status: string): void {
  indicator.className = connected ? 'state-pill connected' : 'state-pill'
  indicator.textContent = connected ? 'Viewer connected' : 'Disconnected'
  startButton.disabled = connected
  stopButton.disabled = !connected
  message.textContent = status
}
function stopPoll(): void {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}
function disconnect(): void {
  stopPoll()
  if (socket !== null) {
    socket.disconnect()
    socket.removeAllListeners()
    socket = null
  }
  clearCurrent('Disconnected · no historical pose is valid')
  connectionState(false, 'Disconnected.')
}
function loopbackOrigin(value: string): string {
  const url = new URL(value)
  if (url.protocol !== 'http:' || !['localhost','127.0.0.1'].includes(url.hostname)
      || !url.port || url.username || url.password || url.pathname !== '/'
      || url.search || url.hash) {
    throw new Error('Only an explicit localhost HTTP observer URL is allowed.')
  }
  return url.origin
}

startButton.addEventListener('click', () => {
  disconnect()
  let origin: string
  try {
    origin = loopbackOrigin(serverUrl.value.trim())
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : 'Invalid observer URL'
    return
  }
  const secret = token.value
  token.value = ''
  if (secret.length < 24 || secret.length > 256) {
    message.textContent = 'Enter the 24–256 character local read-only viewer secret.'
    return
  }
  const client = io(origin + NS, {
    auth: {token: secret},
    transports: ['websocket'],
    reconnection: true,
    reconnectionAttempts: 3,
    timeout: 5000,
    autoConnect: false,
  })
  socket = client
  client.on('connect', () => {
    // All assets and frames arrive in bounded poll acknowledgements.
    // Reconnect always starts with cleared display state and fresh assets.
    clearCurrent('Awaiting a fresh AssetDescription')
    connectionState(true, 'Authorized viewer connected.')
    stopPoll()
    pollTimer = window.setInterval(() => {
      // The watchdog runs even when a poll acknowledgement never arrives.
      hidePoseIfReceiveStreamIsStale()
      client.emit('pose:poll', {}, (reply: unknown) => {
        if (isObject(reply) && reply.ok === true && Array.isArray(reply.messages)) {
          for (const packet of reply.messages) acceptEnvelope(packet)
          if (isNonnegativeInt(reply.dropped_frames)) dropped = reply.dropped_frames
          dropsElement.textContent = String(dropped) + ' / ' + String(gaps)
        }
        hidePoseIfReceiveStreamIsStale()
      })
    }, 250)
  })
  client.on('connect_error', () => {
    clearCurrent('Authentication or local transport failed')
    connectionState(false, 'Local observer connection failed or token rejected.')
  })
  client.on('disconnect', () => {
    stopPoll()
    clearCurrent('Disconnected; all poses invalid')
    connectionState(false, 'Observer disconnected; reconnect for new assets.')
  })
  client.connect()
})
stopButton.addEventListener('click', disconnect)
connectionState(false, 'No session is connected.')
