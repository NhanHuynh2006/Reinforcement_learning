#!/usr/bin/env python3
import os
import argparse
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

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


def make_env(args, logdir):
    def _thunk():
        if not rclpy.ok():
            rclpy.init(args=None)
        env = LaneFollowerSegEnv(
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
        return Monitor(env, filename=os.path.join(logdir, "monitor.csv"))
    return _thunk


def resolve_resume_path(path: str, logdir: str) -> str:
    if path:
        return path
    env_path = os.environ.get("RESUME_PATH", "")
    if env_path:
        return env_path
    # default to final model in logdir if present
    default_path = os.path.join(logdir, "ppo_lane_seg_final.zip")
    if os.path.isfile(default_path):
        return default_path
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default=os.path.expanduser("~/runs/lane_rl_seg"))
    parser.add_argument("--cam-topic", default="/camera/image_raw")
    parser.add_argument("--cmd-vel-topic", default="/diff_cont/cmd_vel_unstamped")
    parser.add_argument("--max-v", type=float, default=0.5)
    parser.add_argument("--max-w", type=float, default=1.5)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--obs-size", type=int, default=120)
    parser.add_argument("--yolo-weights", default="")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=160)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-every", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Resume training from a saved model")
    parser.add_argument("--resume-path", default="", help="Path to .zip model to resume")
    parser.add_argument("--show", action="store_true", help="Show segmentation mask window")
    parser.add_argument("--publish-debug", action="store_true", help="Publish segmentation mask as ROS image")
    parser.add_argument("--debug-topic", default="/segmentation/debug")
    parser.add_argument("--debug-every", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=200_000)
    args = parser.parse_args()
    args.yolo_weights = resolve_yolo_weights(args.yolo_weights)
    resume_env = os.environ.get("RESUME_TRAIN", "").lower() in ("1", "true", "yes", "on")
    args.resume = args.resume or resume_env
    args.resume_path = resolve_resume_path(args.resume_path, args.logdir)

    os.makedirs(args.logdir, exist_ok=True)

    env = DummyVecEnv([make_env(args, args.logdir)])
    env = VecTransposeImage(env)

    if args.resume:
        if not args.resume_path or not os.path.isfile(args.resume_path):
            raise FileNotFoundError(
                f"Resume enabled but model not found: {args.resume_path}"
            )
        model = PPO.load(args.resume_path, env=env, device="cuda")
        print(f"[resume] Loaded: {args.resume_path}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=1.5e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            clip_range=0.2,
            tensorboard_log=args.logdir,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000 // env.num_envs,
        save_path=args.logdir,
        name_prefix="ppo_lane_seg",
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=checkpoint_cb,
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )
    model.save(os.path.join(args.logdir, "ppo_lane_seg_final"))

    try:
        env.envs[0].close()
    except Exception:
        pass



if __name__ == "__main__":
    main()
