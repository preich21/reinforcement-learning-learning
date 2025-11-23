# train_dqn.py
from stable_baselines3 import DQN
from env_flappy import FlappyBirdEnv

def make_env():
    return FlappyBirdEnv()

if __name__ == "__main__":
    env = make_env()

    policy_kwargs = dict(net_arch=[128, 128])  # bigger MLP

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,              # every 4 steps
        gradient_steps=1,
        target_update_interval=5_000,
        exploration_fraction=0.3,  # decays epsilon over first 20% of steps
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu",  # force CPU
        tensorboard_log="./logs_flappy/",
    )

    model.learn(total_timesteps=500_000)
    model.save("dqn_flappy")
