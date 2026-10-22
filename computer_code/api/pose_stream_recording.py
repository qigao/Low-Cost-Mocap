"""Bounded immutable PoseStream JSONL recording and deterministic replay.

A chained SHA-256 catches accidental edits, truncation and reordering of a
recording. It is NOT a cryptographic signature, external ground truth, sensor
exposure timing proof or hardware-calibration certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator

from pose_stream_broker import PoseStreamConsumer, _bounded_positive, _json_size


_GENESIS = "0" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _chain_digest(previous: str, envelope: dict[str, Any]) -> str:
    return sha256(
        (previous + "\n" + _canonical_json(envelope)).encode("utf-8")
    ).hexdigest()


def _reject_duplicate_keys(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise ValueError(f"duplicate recording JSON key {key}")
        obj[key] = value
    return obj


def _invalid_constant(value):
    raise ValueError(f"nonfinite recording JSON constant {value}")


@dataclass(frozen=True)
class PoseStreamLogStatistics:
    recorded_messages: int
    recorded_bytes: int
    head_sha256: str
    hardware_provenance_verified: bool = False


class BoundedPoseStreamJournal:
    """Append-only recording with strict capacity limits and consumer checks.

    Internal storage contains only canonical JSON lines, never live camera
    frames or NumPy-owned runtime data. A rejected append cannot advance the
    journal sequence or independent consumer. A full recording fails closed
    rather than silently losing middle messages or an asset description.
    """

    def __init__(
        self,
        *,
        max_messages: int = 10000,
        max_bytes: int = 32_000_000,
    ) -> None:
        self.max_messages = _bounded_positive(
            max_messages, "max_messages", maximum=1_000_000
        )
        self.max_bytes = _bounded_positive(
            max_bytes, "max_bytes", maximum=1_000_000_000
        )
        self._lines: list[str] = []
        self._total_bytes = 0
        self._head = _GENESIS
        self._consumer = PoseStreamConsumer(
            max_message_bytes=min(self.max_bytes, 100_000_000)
        )

    @property
    def statistics(self) -> PoseStreamLogStatistics:
        return PoseStreamLogStatistics(
            recorded_messages=len(self._lines),
            recorded_bytes=self._total_bytes,
            head_sha256=self._head,
            hardware_provenance_verified=False,
        )

    def append(self, envelope: object) -> str:
        """Validate and add exactly one canonical, hash-chained envelope."""
        if not isinstance(envelope, dict):
            raise ValueError("PoseStream journal requires a wire envelope object")
        if len(self._lines) >= self.max_messages:
            raise ValueError("PoseStream journal message capacity exceeded")
        # Serialize before mutating the independent reader. Unknown object
        # types, NaN and Infinity are rejected by finite JSON serialization.
        try:
            packet = json.loads(_canonical_json(envelope))
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("PoseStream journal envelope is not finite JSON") from exc
        digest = _chain_digest(self._head, packet)
        record = {
            "record_index": len(self._lines),
            "previous_sha256": self._head,
            "sha256": digest,
            "envelope": packet,
        }
        line = _canonical_json(record) + "\n"
        cost = len(line.encode("utf-8"))
        if self._total_bytes + cost > self.max_bytes:
            raise ValueError("PoseStream journal byte capacity exceeded")
        # Important order: consumer validation first, then irreversible log
        # mutation. A malformed/stale envelope cannot enter the recording.
        self._consumer.consume(packet)
        self._lines.append(line)
        self._total_bytes += cost
        self._head = digest
        return digest

    def jsonl(self) -> str:
        return "".join(self._lines)

    def write_jsonl(self, path: str | Path) -> None:
        Path(path).write_text(self.jsonl(), encoding="utf-8")

    def iter_envelopes(self) -> Iterator[dict[str, Any]]:
        """Yield detached JSON objects with the original accepted ordering."""
        for line in self._lines:
            yield json.loads(line)["envelope"]

    @classmethod
    def from_jsonl(
        cls,
        raw: str,
        *,
        max_messages: int = 10000,
        max_bytes: int = 32_000_000,
    ) -> "BoundedPoseStreamJournal":
        if not isinstance(raw, str):
            raise ValueError("PoseStream replay input must be UTF-8 text")
        limit = _bounded_positive(max_bytes, "max_bytes", maximum=1_000_000_000)
        if len(raw.encode("utf-8")) > limit:
            raise ValueError("PoseStream replay exceeds byte capacity")
        result = cls(max_messages=max_messages, max_bytes=max_bytes)
        if not raw:
            return result
        if not raw.endswith("\n"):
            raise ValueError("PoseStream JSONL requires a terminal newline")
        lines = raw.splitlines()
        if len(lines) > result.max_messages:
            raise ValueError("PoseStream replay exceeds message capacity")
        for index, line in enumerate(lines):
            if not line.strip():
                raise ValueError("PoseStream recording contains blank record")
            try:
                data = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_invalid_constant,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("malformed PoseStream JSONL record") from exc
            if not isinstance(data, dict) or set(data) != {
                "record_index", "previous_sha256", "sha256", "envelope",
            }:
                raise ValueError("record keys do not match PoseStream integrity schema")
            if (
                type(data["record_index"]) is not int
                or data["record_index"] != index
                or data["previous_sha256"] != result._head
            ):
                raise ValueError("record order or previous checksum chain was altered")
            packet = data["envelope"]
            if not isinstance(packet, dict):
                raise ValueError("recorded envelope must be an object")
            digest = _chain_digest(result._head, packet)
            if digest != data["sha256"]:
                raise ValueError("PoseStream recording SHA-256 chain mismatch")
            accepted = result.append(packet)
            if accepted != digest:
                raise ValueError("PoseStream journal replay changed checksum")
        return result

    @classmethod
    def read_jsonl(
        cls,
        path: str | Path,
        *,
        max_messages: int = 10000,
        max_bytes: int = 32_000_000,
    ) -> "BoundedPoseStreamJournal":
        file = Path(path)
        max_allowed = _bounded_positive(
            max_bytes, "max_bytes", maximum=1_000_000_000
        )
        if file.stat().st_size > max_allowed:
            raise ValueError("PoseStream recording file exceeds byte capacity")
        return cls.from_jsonl(
            file.read_text(encoding="utf-8"),
            max_messages=max_messages,
            max_bytes=max_bytes,
        )
