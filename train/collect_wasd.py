#!/usr/bin/env python3
import os
import sys
import time
import argparse
import select
import termios
import tty

import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def get_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


class WASDCollector(Node):
    def __init__(self, cam_topic: str, cmd_topic: str) -> None:
        super().__init__("wasd_collector")
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.image_sub = self.create_subscription(Image, cam_topic, self.image_cb, 10)
        self.latest_img = None
        self.latest_stamp = None
        self.frame_count = 0

    def image_cb(self, msg: Image) -> None:
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_stamp = time.time()
            self.frame_count += 1
        except Exception as exc:
            self.get_logger().warn(f"Image convert error: {exc}")

    def publish_cmd(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-topic", default="/camera/image_raw")
    parser.add_argument("--cmd-vel-topic", default="/diff_cont/cmd_vel_unstamped")
    parser.add_argument("--out", default=os.path.expanduser("~/datasets/road_seg/images/train"))
    parser.add_argument("--prefix", default="img")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--every", type=int, default=0, help="Save every N frames (overrides --fps when >0)")
    parser.add_argument("--lin", type=float, default=0.4)
    parser.add_argument("--ang", type=float, default=1.2)
    parser.add_argument("--stop-timeout", type=float, default=0.25)
    parser.add_argument("--only-when-moving", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not rclpy.ok():
        rclpy.init(args=None)
    node = WASDCollector(args.cam_topic, args.cmd_vel_topic)

    v = 0.0
    w = 0.0
    last_key = 0.0
    last_save = 0.0
    last_saved_frame = -1
    idx = 0
    record = True

    print(
        "WASD drive | Space=stop | P=toggle record | Q=quit\n"
        f"Saving to: {args.out}",
        flush=True,
    )

    with RawTerminal():
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.0)
                key = get_key()
                now = time.time()
                if key:
                    last_key = now
                    k = key.lower()
                    if k == "w":
                        v = args.lin
                    elif k == "s":
                        v = -args.lin
                    elif k == "a":
                        w = args.ang
                    elif k == "d":
                        w = -args.ang
                    elif k == " ":
                        v = 0.0
                        w = 0.0
                    elif k == "p":
                        record = not record
                        print(f"[record] {record}", flush=True)
                    elif k == "q":
                        break

                if now - last_key > args.stop_timeout:
                    v = 0.0
                    w = 0.0

                node.publish_cmd(v, w)

                if record and node.latest_img is not None:
                    if args.only_when_moving and abs(v) < 1e-3 and abs(w) < 1e-3:
                        pass
                    else:
                        if args.every and args.every > 0:
                            if node.frame_count != last_saved_frame and (node.frame_count % args.every) == 0:
                                idx += 1
                                fname = os.path.join(args.out, f"{args.prefix}_{idx:06d}.jpg")
                                cv2.imwrite(fname, node.latest_img)
                                last_saved_frame = node.frame_count
                        else:
                            if now - last_save >= (1.0 / max(args.fps, 1e-3)):
                                idx += 1
                                fname = os.path.join(args.out, f"{args.prefix}_{idx:06d}.jpg")
                                cv2.imwrite(fname, node.latest_img)
                                last_save = now

                time.sleep(0.01)
        finally:
            node.publish_cmd(0.0, 0.0)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
