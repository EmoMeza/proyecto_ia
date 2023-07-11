import asyncio
import numpy as np
import sys

from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
from rl.agents.sarsa import SarsaAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate

import tensorflow as tf
import keras

from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import GenData
from poke_env.player import (
    background_evaluate_player,
    evaluate_player,
    background_cross_evaluate,
    cross_evaluate,
    Gen8EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    ObservationType,
    wrap_for_old_gym_api,
    SimpleHeuristicsPlayer,
)


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart
                )

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

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
    opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(battle_format="gen8randombattle", start_challenging=True, opponent=opponent)
    check_env(test_env)
    test_env.close()

    opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    train_env = wrap_for_old_gym_api(train_env)
    opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    eval_env = wrap_for_old_gym_api(eval_env)

    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    model = Sequential()
    model.add(Dense(128, name="Initial", activation="elu", input_shape=input_shape, kernel_initializer='he_uniform', use_bias=False))
    model.add(Flatten())
    model.add(Dense(64, name="Middle", activation="elu", kernel_initializer='he_uniform'))
    model.add(Dense(n_action, activation="linear", name="Output", kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)))

    memory = SequentialMemory(limit=50000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=50000,
    )

    sarsa = SarsaAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=10000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
    )

    sarsa.compile(optimizer=Adam(learning_rate=0.0005), metrics=["mae"])

    # sarsa.load_weights('weights/heur_50k_sarsa_weights.h5f')  # Load weights

    sarsa.fit(train_env, nb_steps=50000)  # Train the SARSA agent

    sarsa.save_weights('weights/heur_50k_sarsa_weights.h5f', overwrite=True)  # Save weights

    train_env.close()
    print("Training done and saved.")

    print("Results against random player:")
    sarsa.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"SARSA Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    sarsa.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"SARSA Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    eval_env.reset_env(restart=False)

    eval_env.reset_env(restart=True, opponent=opponent)

    n_challenges = 100
    placement_battles = 40

    evaluation_future = evaluate_player(
        player=eval_env.agent,
        n_battles=n_challenges,
        n_placement_battles=placement_battles,
    )
    print("Evaluation with included method:", evaluation_future)

    eval_env.reset_env(restart=False)

    n_challenges = 100
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    sarsa.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=True,
        visualize=True,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of SARSA with baselines:")
    print(tabulate(table))

    eval_env.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
