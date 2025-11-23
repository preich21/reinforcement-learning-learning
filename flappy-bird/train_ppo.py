from stable_baselines3 import PPO
from env_flappy import FlappyBirdEnv

def make_env():
    return FlappyBirdEnv()

if __name__ == "__main__":
    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        verbose=1,
        device="cpu",
        tensorboard_log="./logs_flappy/",
    )

    model.learn(total_timesteps=1_000_000)
    model.save("ppo_flappy")
