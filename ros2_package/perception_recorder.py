#!/usr/bin/env python3
"""
Perception Recorder Node - saves the Perception output to disk.

The node publishes PerceptionArray on a topic and nothing else persists it.
This subscribes and writes two files:

  <out>.jsonl  one JSON object per PerceptionArray message, full fidelity
  <out>.csv    one row per perception, flat, for a quick eyeball or a plot

Field access goes through getattr so the same recorder works whether or not
the built msgs package defines the optional frame_number, occlusion and pose
fields.

Usage:
    python3 perception_recorder.py --output /output/perceptions
"""

import argparse
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

from msgs.msg import PerceptionArray

DEFAULT_TOPIC = "/vanderbilt/fake_perception/data"

# Must match the publisher: prediction_node subscribes with this too.
PERCEPTION_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

CSV_COLUMNS = [
    "stamp_sec", "frame_num", "tracking_id", "target_entity_id",
    "detection_prob", "north_m", "east_m", "down_m", "yaw",
    "entity_class", "entity_color", "match_prob",
]


class PerceptionRecorder(Node):

    def __init__(self, output_stem: str, topic: str):
        super().__init__("perception_recorder")

        self.jsonl_path = Path(f"{output_stem}.jsonl")
        self.csv_path = Path(f"{output_stem}.csv")
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self.jsonl = open(self.jsonl_path, "w")
        self.csv = open(self.csv_path, "w")
        self.csv.write(",".join(CSV_COLUMNS) + "\n")

        self.msg_count = 0
        self.row_count = 0
        self.matched_count = 0

        self.create_subscription(
            PerceptionArray, topic, self.callback, PERCEPTION_QOS)

        self.get_logger().info(f"Recording {topic}")
        self.get_logger().info(f"  -> {self.jsonl_path}")
        self.get_logger().info(f"  -> {self.csv_path}")

    def callback(self, msg: PerceptionArray):
        stamp = msg.stamp.sec + msg.stamp.nanosec * 1e-9

        record = {
            "stamp": stamp,
            "frame_num": int(msg.frame_num),
            "perceptions": [],
        }

        for p in msg.perceptions:
            location = [float(v) for v in p.location]
            entry = {
                "tracking_id": int(p.tracking_id),
                "target_entity_id": p.target_entity_id,
                "detection_prob": float(p.detection_prob),
                "location_ned_m": location,
                "yaw": float(p.yaw),
                "entity_class": p.entity_class,
                "entity_color": p.entity_color,
                "match_prob": float(p.match_prob),
            }
            # Optional; absent from this repo's definitions.
            for field in ("frame_number", "occlusion", "pose"):
                if hasattr(p, field):
                    entry[field] = getattr(p, field)
            record["perceptions"].append(entry)

            self.csv.write(",".join([
                f"{stamp:.3f}", str(msg.frame_num), str(p.tracking_id),
                p.target_entity_id, f"{p.detection_prob:.4f}",
                f"{location[0]:.3f}", f"{location[1]:.3f}", f"{location[2]:.3f}",
                f"{p.yaw:.4f}", p.entity_class, p.entity_color,
                f"{p.match_prob:.4f}",
            ]) + "\n")
            self.row_count += 1
            if p.target_entity_id:
                self.matched_count += 1

        self.jsonl.write(json.dumps(record) + "\n")
        self.jsonl.flush()
        self.csv.flush()

        self.msg_count += 1
        if self.msg_count % 30 == 0:
            self.get_logger().info(
                f"{self.msg_count} messages, {self.row_count} perceptions, "
                f"{self.matched_count} with a target_entity_id")

    def close(self):
        self.jsonl.close()
        self.csv.close()
        print(f"\nRecorded {self.msg_count} messages / {self.row_count} "
              f"perceptions ({self.matched_count} matched to an entity)")
        print(f"  {self.jsonl_path}")
        print(f"  {self.csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="/output/perceptions",
                        help="output path stem; .jsonl and .csv are appended")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    args = parser.parse_args()

    rclpy.init()
    node = PerceptionRecorder(args.output, args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
