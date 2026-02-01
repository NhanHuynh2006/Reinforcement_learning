#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from seg_env import LaneFollowerSegEnv

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "yolo.pt")


def resolve_yolo_weights(path: str) -> str:
    if path:
        return path
    env_path = os.environ.get("YOLO_WEIGHTS", "")
    if env_path:
        return env_path
    if os.path.isfile(DEFAULT_WEIGHTS):
        return DEFAULT_WEIGHTS
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cam-topic", default="/camera/image_raw")
    parser.add_argument("--cmd-vel-topic", default="/diff_cont/cmd_vel_unstamped")
    parser.add_argument("--max-v", type=float, default=0.5)
    parser.add_argument("--max-w", type=float, default=1.5)
    parser.add_argument("--step-dt", type=float, default=0.1)
    parser.add_argument("--obs-size", type=int, default=120)
    parser.add_argument("--yolo-weights", default="")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=320)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-every", type=int, default=1)
    parser.add_argument("--show", action="store_true", help="Show segmentation mask window")
    parser.add_argument("--publish-debug", action="store_true", help="Publish segmentation mask as ROS image")
    parser.add_argument("--debug-topic", default="/segmentation/debug")
    parser.add_argument("--debug-every", type=int, default=5)
    args = parser.parse_args()
    args.yolo_weights = resolve_yolo_weights(args.yolo_weights)

    rclpy.init()

    def _make_env():
        return LaneFollowerSegEnv(
            cam_topic=args.cam_topic,
            cmd_vel_topic=args.cmd_vel_topic,
            max_v=args.max_v,
            max_w=args.max_w,
            step_dt=args.step_dt,
            obs_size=args.obs_size,
            yolo_weights=args.yolo_weights,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_imgsz=args.yolo_imgsz,
            yolo_device=args.yolo_device,
            yolo_every=args.yolo_every,
            show_debug=args.show,
            debug_every=args.debug_every,
            publish_debug=args.publish_debug,
            debug_topic=args.debug_topic,
        )

    env = DummyVecEnv([_make_env])
    env = VecTransposeImage(env)

    model = PPO.load(args.model, env=env)
    obs = env.reset()
    try:
        while rclpy.ok():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if isinstance(done, (list, tuple, np.ndarray)):
                if np.any(done):
                    obs = env.reset()
            elif done:
                obs = env.reset()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
