import asyncio
import numpy as np
import random
from poke_env.player import cross_evaluate, Gen8EnvSinglePlayer, RandomPlayer, wrap_for_old_gym_api
from tabulate import tabulate
from threading import Thread
import keras
import json
from poke_env import to_id_str
from poke_env.player import Player
from poke_env.player_configuration import PlayerConfiguration
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from poke_env.data import GenData

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import ObservationType
from gym.spaces import Space, Box

from poke_env import ShowdownServerConfiguration


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart = GenData.from_gen(7).type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


async def main():
    # Create an instance of the SimpleRLPlayer
    
    bot_team = await get_bot_team()
    rl_player = SimpleRLPlayer(battle_format="gen8randombattle", opponent="Dogepot",server_configuration=ShowdownServerConfiguration, player_configuration=rl_config)

    # Create the environment for training
    train_env = wrap_for_old_gym_api(rl_player)

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(128, name="Initial", activation="elu", input_shape=input_shape, kernel_initializer='he_uniform', use_bias=False))
    model.add(Flatten())
    model.add(Dense(64, name="Middle", activation="elu", kernel_initializer='he_uniform'))
    model.add(Dense(n_action, activation="linear", name="Output", kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))

    # Defining the DQN
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
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(optimizer=Adam(learning_rate=0.00025), metrics=["mae"])

    # Load the pre-trained weights for rl_player
    dqn.load_weights('weights/dqn_weights.h5f')
    dqn.test(train_env, nb_episodes=1, verbose=True, visualize=True) 

    train_env.close()



### Functions to create the teams

async def get_bot_team():
    with open('bot_team.json') as json_file:
        data = json.load(json_file)
        return data["team"] 
    

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())