from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dino import DinoEnv


def make_env(rank, base_seed=0):
    def _init():
        env = DinoEnv(render_mode=None)
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env
    return _init

if __name__ == "__main__":
    n_envs = 8
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        device="cuda",
        verbose=1,
        tensorboard_log="./logs_dino/",
    )

    model.learn(total_timesteps=2_000_000)
    model.save("ppo_dino_cnn")
