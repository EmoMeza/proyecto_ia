import numpy as np
from gym.spaces import Box
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import LocalhostServerConfiguration
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.python.keras.optimizer_v2 import adam as Adam
from poke_env.player import wrap_for_old_gym_api

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle):
        # Embedding logic here
        pass

    def describe_embedding(self):
        # Description of the embedding here
        pass

# Instantiate the environment
player = SimpleRLPlayer(battle_format="gen8randombattle")

# Wrap the environment
env = wrap_for_old_gym_api(player)

# Define the model
model = Sequential()
model.add(Dense(128, activation="elu", input_shape=env.observation_space.shape))
model.add(Flatten())
model.add(Dense(64, activation="elu"))
model.add(Dense(env.action_space.n, activation="linear"))

# Define the DQN agent
memory = SequentialMemory(limit=10000, window_length=1)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr="eps",
    value_max=1.0,
    value_min=0.05,
    value_test=0.0,
    nb_steps=10000,
)
dqn = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    policy=policy,
    memory=memory,
    nb_steps_warmup=1000,
    gamma=0.5,
    target_model_update=1,
    delta_clip=0.01,
    enable_double_dqn=True,
)
dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

# Train the model
dqn.fit(env, nb_steps=10000)

# Save the model
dqn.save_weights('dqn_weights.h5f', overwrite=True)
