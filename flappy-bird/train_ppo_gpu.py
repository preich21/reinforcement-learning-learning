from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from env_flappy import FlappyBirdEnv

def make_env(rank, base_seed=0):
    def _init():
        created_env = FlappyBirdEnv(render_mode=None)
        created_env.reset(seed=base_seed + rank)
        return created_env
    return _init

if __name__ == "__main__":
    n_envs = 32
    env = SubprocVecEnv([make_env(i, base_seed=42) for i in range(n_envs)])

    torch .set_num_threads(1)

    policy_kwargs = dict(
        net_arch=[512, 512],    # bigger net = more GPU work
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=8192,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        verbose=1,
        device="cuda",
        tensorboard_log="./logs_flappy/",
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=10_000_000)
    model.save("ppo_flappy")
