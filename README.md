# Reinforcement Learning Flappy Bird

A little learning project for myself to get familiar with Reinforcement Learning.

The `flappy-bird` folder contains a little python-native Flappy Bird like game, on which a PPO agent with the MLP policy
has been trained.

The `dino-game` folder contains a Chrome Offline Dino Game like implementation, with a 64x64 "pixel" UI. A PPO agent
with the CNN policy has been trained on this environment.


## Prerequisites

To run the code, please create a venv first and install the required packages:

```shell
python -m venv .venv
source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```


## Watch the agent play

Included in the two game folders are the pre-trained models `ppo_flappy.zip` and `ppo_dino_cnn.zip`.

To watch the Flappy Bird agent play, run:

```shell
cd flappy-bird # always run the scripts from the game folder
python watch_agent.py ppo
```

To watch the Dino Game agent play, run:

```shell
cd dino-game # always run the scripts from the game folder
python play_dino.py --mode agent
```


## Play the games yourself

You can also play the games yourself!
This helps with getting a feel for the complexity of the tasks.

For the Flappy Bird game, run:

```shell
cd flappy-bird # always run the scripts from the game folder
python play_manual.py
```

Note that the UI differs from the one the agent is modeled with, however, the game mechanics are the same.

For the Dino Game, run:

```shell
cd dino-game # always run the scripts from the game folder
python play_dino.py --mode manual
```


## Train your own agent

To train you own Flappy Bird agent, run:

```shell
cd flappy-bird # always run the scripts from the game folder
python train_ppo.py # Choose one of: train_dqn.py, train_ppo.py or train_ppo_gpu.py
```

To train your own Dino Game agent, run:

```shell
cd dino-game # always run the scripts from the game folder
python train_ppo_gpu.py # For this you'll need a cuda-capable GPU. Otherwise set device="cpu" in the file.
```

