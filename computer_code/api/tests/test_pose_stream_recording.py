"""Deterministic bounded PoseStream journal and independent checksum replay."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from board_patterns import BoardPatternRegistry
from board_recognizer import BoardOutcome, BoardPose
from identity_locked_board_tracker import BoardFrameStatus, LockedBoardFrame
from pose_stream_broker import BoundedPoseStreamPublisher
from pose_stream_recording import BoundedPoseStreamJournal


CONFIG = Path(__file__).resolve().parents[1] / "board-patterns.continuous.example.json"


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(CONFIG)
        self.publisher = BoundedPoseStreamPublisher(
            self.registry, max_pending_frames=8
        )
        self.publisher.subscribe("recorder")

    def frame(self, trigger, *, measured=True, calibration="a"*64):
        board = self.registry.boards["board_A"]
        board_xyz = np.array([0.2 + 0.001 * trigger, 0.0, 1.5])
        t_body = board_xyz - board.translation_body_from_board_m
        pose = BoardPose(
            board_id="board_A", trigger_id=trigger,
            marker_ids=(100,101,102), rotation_world_from_board=np.eye(3),
            translation_world_from_board=board_xyz,
            rotation_world_from_body=np.eye(3),
            translation_world_from_body=t_body,
            rigid_rms_error=0.001, reprojection_max_rms_px=0.07,
        )
        outcome_a = BoardOutcome(
            "board_A", "recognized" if measured else "not_visible",
            (100,101,102) if measured else (), (),
            pose if measured else None,
        )
        outcome_b = BoardOutcome("board_B", "not_visible", (), (), None)
        stamp = 2_000_000 + 1000 * trigger
        a = BoardFrameStatus(
            "board_A", "TRACKING" if measured else "DISCOVER",
            "MEASURED" if measured else "INVALID",
            trigger if measured else None,
            0 if measured else None,
            stamp if measured else None,
            0 if measured else None,
            outcome_a,
        )
        b = BoardFrameStatus(
            "board_B", "DISCOVER", "INVALID",
            None, None, None, None, outcome_b,
        )
        return LockedBoardFrame(
            trigger_id=trigger, camera_ids=(2,5,9),
            clock_domain="external-reference-declared", exposure_skew_ns=9,
            latest_exposure_timestamp_ns=stamp,
            pattern_sha256=self.registry.config_sha256,
            calibration_sha256=calibration,
            boards={"board_A":a,"board_B":b},
            active_locks=(), newly_confirmed_marker_ids=(),
            invalidated_marker_ids=(),
            measured_marker_ids=(100,101,102) if measured else (),
        )

    def collect(self, *, count=3, journal=None):
        if journal is None:
            journal = BoundedPoseStreamJournal()
        self.publisher.publish(self.frame(1))
        for trigger in range(2,count+1):
            self.publisher.publish(self.frame(trigger, measured=trigger%2 == 0))
        while (packet := self.publisher.peek_next("recorder")) is not None:
            journal.append(packet)
            self.publisher.acknowledge("recorder", packet["sequence"])
        return journal

    def test_record_replay_exact_order_checksums_and_hardware_false(self):
        journal=self.collect()
        stats=journal.statistics
        self.assertEqual(stats.recorded_messages,4)
        self.assertEqual(len(stats.head_sha256),64)
        self.assertGreater(stats.recorded_bytes,0)
        self.assertFalse(stats.hardware_provenance_verified)
        envelopes=list(journal.iter_envelopes())
        self.assertEqual([e["payload"]["kind"] for e in envelopes],
                         ["asset_description","pose_frame","pose_frame","pose_frame"])
        self.assertEqual(
            [e["payload"]["frame_id"] for e in envelopes[1:]],
            [1,2,3],
        )
        copied=envelopes[1]
        copied["payload"]["rigid_bodies"].clear()
        self.assertEqual(
            len(list(journal.iter_envelopes())[1]["payload"]["rigid_bodies"]),2
        )
        replay=BoundedPoseStreamJournal.from_jsonl(journal.jsonl())
        self.assertEqual(replay.jsonl(), journal.jsonl())
        self.assertEqual(replay.statistics, stats)

    def test_tampered_geometry_old_record_and_reordered_chain_fail_closed(self):
        tape=self.collect()
        records=[json.loads(x) for x in tape.jsonl().splitlines()]
        tampered=deepcopy(records)
        tampered[1]["envelope"]["payload"]["rigid_bodies"][0][
            "position_world_from_body"
        ][0]+=100
        bad="\n".join(json.dumps(x) for x in tampered)+"\n"
        with self.assertRaisesRegex(ValueError, "SHA-256 chain mismatch"):
            BoundedPoseStreamJournal.from_jsonl(bad)
        reordered=[records[0],records[2],records[1],records[3]]
        with self.assertRaisesRegex(ValueError, "record order|checksum chain"):
            BoundedPoseStreamJournal.from_jsonl(
                "\n".join(json.dumps(x) for x in reordered)+"\n"
            )
        with self.assertRaisesRegex(ValueError, "record order|checksum chain"):
            BoundedPoseStreamJournal.from_jsonl(
                "\n".join(json.dumps(x) for x in records[1:])+"\n"
            )

    def test_bounded_message_and_byte_limit_does_not_advance_recording(self):
        j=BoundedPoseStreamJournal(max_messages=2)
        self.publisher.publish(self.frame(1))
        self.publisher.publish(self.frame(2))
        asset=self.publisher.peek_next("recorder")
        j.append(asset)
        self.publisher.acknowledge("recorder", asset["sequence"])
        one=self.publisher.peek_next("recorder")
        j.append(one)
        self.publisher.acknowledge("recorder", one["sequence"])
        prior=j.statistics
        two=self.publisher.peek_next("recorder")
        with self.assertRaisesRegex(ValueError, "message capacity"):
            j.append(two)
        self.assertEqual(j.statistics,prior)
        # The consumer has not consumed frame 2 after capacity refusal.
        output=BoundedPoseStreamJournal.from_jsonl(j.jsonl())
        self.assertEqual(output.statistics.recorded_messages,2)

        tiny=BoundedPoseStreamJournal(
            max_messages=100,
            max_bytes=len((json.dumps(asset)).encode())+100,
        )
        with self.assertRaisesRegex(ValueError, "byte capacity"):
            tiny.append(asset)
        self.assertEqual(tiny.statistics.recorded_messages,0)

    def test_invalid_payload_does_not_enter_integrity_recording(self):
        journal=BoundedPoseStreamJournal()
        self.publisher.publish(self.frame(1))
        asset=self.publisher.peek_next("recorder")
        journal.append(asset)
        self.publisher.acknowledge("recorder",0)
        real=self.publisher.peek_next("recorder")
        bad=deepcopy(real)
        bad["payload"]["hardware_provenance_verified"]=True
        stats=journal.statistics
        with self.assertRaisesRegex(ValueError, "PoseFrame"):
            journal.append(bad)
        self.assertEqual(journal.statistics,stats)
        journal.append(real)
        self.assertEqual(journal.statistics.recorded_messages,2)

    def test_recording_and_replay_through_disk(self):
        journal=self.collect()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"poseframes.jsonl"
            journal.write_jsonl(path)
            replay=BoundedPoseStreamJournal.read_jsonl(path)
            self.assertEqual(replay.statistics,journal.statistics)
            path.write_text(journal.jsonl()[:-2],encoding="utf8")
            with self.assertRaisesRegex(ValueError, "terminal newline"):
                BoundedPoseStreamJournal.read_jsonl(path)
            path.write_text(journal.jsonl() + "\n",encoding="utf8")
            with self.assertRaisesRegex(ValueError, "blank record"):
                BoundedPoseStreamJournal.read_jsonl(path)

    def test_restarted_session_in_recording_requires_new_asset(self):
        journal=BoundedPoseStreamJournal()
        self.publisher.publish(self.frame(5))
        while (packet:=self.publisher.peek_next("recorder")) is not None:
            journal.append(packet)
            self.publisher.acknowledge("recorder",packet["sequence"])
        self.publisher.restart_session()
        self.publisher.publish(self.frame(1,calibration="b"*64))
        while (packet:=self.publisher.peek_next("recorder")) is not None:
            journal.append(packet)
            self.publisher.acknowledge("recorder",packet["sequence"])
        payloads=[x["payload"]["kind"] for x in journal.iter_envelopes()]
        self.assertEqual(payloads,[
            "asset_description","pose_frame","asset_description","pose_frame"
        ])
        restored=BoundedPoseStreamJournal.from_jsonl(journal.jsonl())
        self.assertEqual(restored.statistics.head_sha256,journal.statistics.head_sha256)

    def test_duplicate_json_keys_and_nonfinite_values_are_rejected(self):
        # Parser detects duplicate keys even if the JSON itself is otherwise
        # syntactically legal.
        with self.assertRaisesRegex(ValueError, "malformed"):
            BoundedPoseStreamJournal.from_jsonl(
                '{"record_index":0,"record_index":0}\n'
            )
        with self.assertRaisesRegex(ValueError, "malformed"):
            BoundedPoseStreamJournal.from_jsonl(
                '{"record_index":0,"previous_sha256":"x","sha256":"x","envelope":NaN}\n'
            )


if __name__=="__main__":
    unittest.main()
