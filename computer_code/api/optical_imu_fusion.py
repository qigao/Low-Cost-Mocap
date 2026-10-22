"""Bounded optical/IMU attitude fusion; orientation only, no flight-control output."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from numbers import Real
from imu_contract import ClockMapping
from imu_propagation import AttitudePropagationState

def _f(x,n):
    if isinstance(x,bool) or not isinstance(x,Real): raise ValueError(f"{n} must be finite real")
    x=float(x)
    if not math.isfinite(x): raise ValueError(f"{n} must be finite real")
    return x
def _t(x,n):
    if type(x) is not int or not 0<=x<2**63: raise ValueError(f"{n} must be a nonnegative int64")
    return x
def _s(x,n):
    if not isinstance(x,str) or not x.strip(): raise ValueError(f"{n} must be nonempty")
    return x.strip()
def _h(x,n):
    if not isinstance(x,str) or len(x)!=64 or any(c not in "0123456789abcdef" for c in x): raise ValueError(f"{n} must be lowercase SHA-256")
    return x
def _q(x):
    if not isinstance(x,(tuple,list)) or len(x)!=4: raise ValueError("quaternion must be XYZW")
    q=tuple(_f(v,"quaternion") for v in x); n=math.sqrt(sum(v*v for v in q))
    if n<1e-12: raise ValueError("quaternion norm is zero")
    return tuple(v/n for v in q)
def _mul(a,b):
    x,y,z,w=a; X,Y,Z,W=b
    return (w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z)
def _inv(q): return (-q[0],-q[1],-q[2],q[3])
def _log(q):
    q=_q(q)
    if q[3]<0: q=tuple(-x for x in q)
    n=math.sqrt(sum(x*x for x in q[:3]))
    if n<1e-12: return (0.,0.,0.)
    a=2*math.atan2(n,max(-1.,min(1.,q[3])))
    return tuple(q[i]*a/n for i in range(3))
def _exp(v):
    a=math.sqrt(sum(x*x for x in v))
    if a<1e-12: return _q((v[0]/2,v[1]/2,v[2]/2,1.))
    k=math.sin(a/2)/a
    return (v[0]*k,v[1]*k,v[2]*k,math.cos(a/2))
def _interp(a,b,f): return _q(_mul(a,_exp(tuple(f*x for x in _log(_mul(_inv(a),b))))))

@dataclass(frozen=True)
class OpticalAttitudeMeasurement:
    body_id: str
    timestamp_ns: int
    clock_domain: str
    quaternion_world_from_body_xyzw: tuple[float,float,float,float]
    frame_id: int
    pattern_sha256: str
    calibration_sha256: str
    quality: str = "MEASURED"
    def __post_init__(self):
        object.__setattr__(self,"body_id",_s(self.body_id,"body_id"))
        object.__setattr__(self,"timestamp_ns",_t(self.timestamp_ns,"timestamp"))
        object.__setattr__(self,"clock_domain",_s(self.clock_domain,"clock domain"))
        object.__setattr__(self,"quaternion_world_from_body_xyzw",_q(self.quaternion_world_from_body_xyzw))
        object.__setattr__(self,"frame_id",_t(self.frame_id,"frame_id"))
        _h(self.pattern_sha256,"pattern digest"); _h(self.calibration_sha256,"calibration digest")
        if self.quality!="MEASURED": raise ValueError("optical measurement must be current MEASURED data")

@dataclass(frozen=True)
class OpticalIMUFusionState:
    body_id: str
    imu_id: str
    timestamp_ns: int
    clock_domain: str
    quaternion_world_from_body_xyzw: tuple[float,float,float,float] | None
    attitude_variance_rad2: float | None
    status: str
    source: str
    last_optical_timestamp_ns: int | None
    last_optical_frame_id: int | None
    optical_age_ns: int | None
    clock_mapping_sha256: str
    imu_calibration_sha256: str
    global_position_m: None = None
    global_velocity_m_s: None = None
    global_position_observable: bool = False
    def __post_init__(self):
        if self.status not in {"TRACKING","DEGRADED","LOST"}: raise ValueError("invalid fusion status")
        if self.status=="LOST":
            if self.quaternion_world_from_body_xyzw is not None or self.attitude_variance_rad2 is not None: raise ValueError("LOST must not publish pose")
        else:
            if self.quaternion_world_from_body_xyzw is None or self.attitude_variance_rad2 is None: raise ValueError("valid state requires pose and variance")
            object.__setattr__(self,"quaternion_world_from_body_xyzw",_q(self.quaternion_world_from_body_xyzw))
            v=_f(self.attitude_variance_rad2,"variance")
            if v<0: raise ValueError("variance must be nonnegative")
        if self.global_position_m is not None or self.global_velocity_m_s is not None or self.global_position_observable is not False: raise ValueError("attitude fusion cannot publish global position")
        _h(self.clock_mapping_sha256,"mapping digest"); _h(self.imu_calibration_sha256,"IMU calibration digest")

class OpticalIMUAttitudeFusion:
    """Isotropic small-angle covariance update with bounded interpolation history."""
    def __init__(self, *, body_id, imu_id, imu_clock_domain, imu_calibration_revision, imu_calibration_sha256,
        optical_pattern_sha256, optical_calibration_sha256, clock_mapping: ClockMapping,
        gyro_noise_density_rad_s_sqrt_hz, optical_variance_rad2=0.0025, max_dropout_ns=100_000_000,
        max_innovation_rad=1.2, history_capacity=256):
        self.body=_s(body_id,"body_id"); self.imu=_s(imu_id,"imu_id"); self.clock=_s(imu_clock_domain,"IMU clock")
        self.calrev=_s(imu_calibration_revision,"IMU calibration revision"); self.calsha=_h(imu_calibration_sha256,"IMU calibration digest")
        self.pattern=_h(optical_pattern_sha256,"pattern digest"); self.optcal=_h(optical_calibration_sha256,"optical calibration digest")
        if not isinstance(clock_mapping,ClockMapping) or clock_mapping.target_clock_domain!=self.clock: raise ValueError("clock mapping target mismatch")
        self.mapping=clock_mapping; self.mapsha=_h(clock_mapping.mapping_sha256,"mapping digest")
        if not isinstance(gyro_noise_density_rad_s_sqrt_hz,(tuple,list)) or len(gyro_noise_density_rad_s_sqrt_hz)!=3: raise ValueError("gyro noise must be xyz")
        self.noise=tuple(_f(x,"gyro noise") for x in gyro_noise_density_rad_s_sqrt_hz)
        if any(x<0 for x in self.noise): raise ValueError("gyro noise must be nonnegative")
        self.r=_f(optical_variance_rad2,"optical variance")
        if self.r<=0: raise ValueError("optical variance must be positive")
        self.dropout=_t(max_dropout_ns,"dropout limit"); self.innovation=_f(max_innovation_rad,"innovation limit")
        if self.dropout<1 or not 0<self.innovation<=math.pi: raise ValueError("invalid dropout/innovation limit")
        if type(history_capacity) is not int or not 3<=history_capacity<=4096: raise ValueError("history capacity must be 3..4096")
        self.history=deque(maxlen=history_capacity); self.q=None; self.p=None; self.last_optical=None; self.last_frame=None; self.expired=False; self.state=None
    def _state(self,ts,status,source):
        age=None if self.last_optical is None else ts-self.last_optical
        return OpticalIMUFusionState(self.body,self.imu,ts,self.clock,self.q if status!="LOST" else None,self.p if status!="LOST" else None,status,source,self.last_optical,self.last_frame,age,self.mapsha,self.calsha)
    def ingest_imu(self,s: AttitudePropagationState):
        if not isinstance(s,AttitudePropagationState): raise ValueError("expected attitude propagation state")
        if s.imu_id!=self.imu or s.clock_domain!=self.clock: raise ValueError("IMU identity/clock mismatch")
        if s.calibration_revision!=self.calrev or s.calibration_sha256!=self.calsha: raise ValueError("IMU calibration mismatch")
        if self.history and (s.sequence_number<=self.history[-1].sequence_number or s.sensor_timestamp_ns<=self.history[-1].sensor_timestamp_ns): raise ValueError("IMU state is duplicate or out of order")
        if self.history:
            prev=self.history[-1]; delta=_mul(_inv(prev.orientation_world_from_body_xyzw),s.orientation_world_from_body_xyzw)
            if self.q is not None:
                self.q=_q(_mul(self.q,delta)); self.p+=max(self.noise)**2*((s.sensor_timestamp_ns-prev.sensor_timestamp_ns)/1e9)
        self.history.append(s)
        if self.last_optical is None or self.q is None: self.state=self._state(s.sensor_timestamp_ns,"LOST","LOST")
        elif s.sensor_timestamp_ns-self.last_optical>self.dropout:
            self.q=None; self.p=None; self.expired=True; self.state=self._state(s.sensor_timestamp_ns,"LOST","LOST")
        else:
            status="TRACKING" if s.sensor_timestamp_ns==self.last_optical else "DEGRADED"
            self.state=self._state(s.sensor_timestamp_ns,status,"OPTICAL_CORRECTED" if status=="TRACKING" else "IMU_PROPAGATED")
        return self.state
    def ingest_optical(self,m: OpticalAttitudeMeasurement, *, allow_reseed_after_lost=False):
        if not isinstance(m,OpticalAttitudeMeasurement): raise ValueError("expected optical measurement")
        if m.body_id!=self.body or m.pattern_sha256!=self.pattern or m.calibration_sha256!=self.optcal: raise ValueError("optical identity/calibration mismatch")
        if m.clock_domain!=self.mapping.source_clock_domain: raise ValueError("optical clock mapping source mismatch")
        t=self.mapping.map_timestamp(m.timestamp_ns)
        if not self.history: raise ValueError("IMU history not initialized")
        if self.last_optical is not None and (t<=self.last_optical or m.frame_id<=self.last_frame): raise ValueError("optical measurement is duplicate/out of order")
        h=list(self.history); times=[x.sensor_timestamp_ns for x in h]
        if t<times[0] or t>times[-1]: raise ValueError("optical timestamp outside retained IMU history")
        hi=next((i for i,x in enumerate(times) if x>=t),len(times)-1)
        if times[hi]==t: q_at=h[hi].orientation_world_from_body_xyzw; bracket=0
        else:
            lo=hi-1; bracket=times[hi]-times[lo]; q_at=_interp(h[lo].orientation_world_from_body_xyzw,h[hi].orientation_world_from_body_xyzw,(t-times[lo])/bracket)
        cur=h[-1]; after=_mul(_inv(q_at),cur.orientation_world_from_body_xyzw); dt=(cur.sensor_timestamp_ns-t)/1e9
        if self.q is None:
            if self.expired and not allow_reseed_after_lost: raise ValueError("explicit reseed required after optical dropout expiry")
            q_at_fused=m.quaternion_world_from_body_xyzw; p_at=self.r
        else:
            back=_mul(_inv(after),self.q); p_at=max(1e-12,self.p-max(self.noise)**2*dt)
            err=_log(_mul(_inv(back),m.quaternion_world_from_body_xyzw)); angle=math.sqrt(sum(x*x for x in err))
            if angle>self.innovation: raise ValueError("optical attitude innovation exceeds limit")
            uncertainty=(self.mapping.observed_max_residual_ns+bracket/2)/1e9
            r_eff=self.r+(max(self.noise)*uncertainty)**2
            gain=p_at/(p_at+r_eff)
            q_at_fused=_q(_mul(back,_exp(tuple(gain*x for x in err))))
            p_at=(1-gain)*p_at
        self.q=_q(_mul(q_at_fused,after)); self.p=p_at+max(self.noise)**2*dt
        self.last_optical=t; self.last_frame=m.frame_id; self.expired=False
        status="TRACKING" if t==cur.sensor_timestamp_ns else "DEGRADED"
        self.state=self._state(cur.sensor_timestamp_ns,status,"OPTICAL_CORRECTED" if status=="TRACKING" else "OPTICAL_CORRECTED_WITH_ALIGNMENT")
        return self.state
