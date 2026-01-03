import argparse
import time

import cv2
from stable_baselines3 import PPO

from env_dino import DinoEnv


def make_env(render_mode="rgb_array"):
    return DinoEnv(render_mode=render_mode)


def show_frame(frame, window_name="Dino"):
    """
    frame: (H, W, 1) uint8 (grayscale) from env
    We'll upscale and convert to BGR for nicer viewing.
    """
    # Remove channel dimension, convert to 3-channel
    frame_gray = frame[:, :, 0]
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Optional: scale up for visibility
    frame_bgr = cv2.resize(frame_bgr, (336, 336), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, frame_bgr)


def run_manual():
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()
    window_name = "Dino - Manual (SPACE to jump, Q to quit)"
    cv2.namedWindow(window_name)

    print("Manual mode: press SPACE to jump, 'q' to quit.")

    done = False
    episode_reward = 0.0

    while True:
        show_frame(obs, window_name=window_name)
        # waitKey returns 16-bit value, we only care low byte
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # space = jump
        if key == ord(" "):
            action = 1
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        if done:
            print(f"Episode finished. Score={info.get('score')}, "
                  f"Return={episode_reward:.2f}")
            episode_reward = 0.0
            obs, info = env.reset()

        # control speed a bit (optional)
        time.sleep(1.0 / 60.0)

    env.close()
    cv2.destroyAllWindows()


def run_agent(model_path: str):
    print(f"Loading agent from: {model_path}")
    model = PPO.load(model_path, device="cpu")  # CPU is enough for inference here

    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()
    window_name = "Dino - Agent (Q to quit)"
    cv2.namedWindow(window_name)

    episode_reward = 0.0

    print("Agent mode: watching agent play. Press 'q' to quit.")

    while True:
        show_frame(obs, window_name=window_name)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Agent action
        action, _ = model.predict(obs, deterministic=True)
        # Stable Baselines3 returns numpy scalar / array, cast to int
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        if done:
            print(f"Episode finished. Score={info.get('score')}, "
                  f"Return={episode_reward:.2f}")
            episode_reward = 0.0
            obs, info = env.reset()

        time.sleep(1.0 / 60.0)

    env.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Dino visual RL: play or watch agent.")
    parser.add_argument(
        "--mode",
        choices=["manual", "agent"],
        required=True,
        help="manual: play yourself; agent: watch trained PPO agent",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_dino_cnn.zip",
        help="Path to trained PPO model (.zip) when using --mode agent",
    )

    args = parser.parse_args()

    if args.mode == "manual":
        run_manual()
    else:
        run_agent(args.model_path)


if __name__ == "__main__":
    main()
