import asyncio
import time
import numpy as np
import json
import random
import poke_env.data
from poke_env.player import Player, RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.player import Player
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon



#Code to create a player that always chooses the move with the highest base power


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: self.damage_multiplier(move, battle.opponent_active_pokemon))
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def damage_multiplier(self, move: Move, opponent: Pokemon):
        # Check if the move is a damaging move
        if move.base_power > 0:
            # Get the type multiplier
            type_multiplier = opponent.damage_multiplier(move.type)
            return move.base_power * type_multiplier
        else:
            return 0
        


class MaxDefensePlayer(Player):
    last_pokemon = ""
    protected = False

    def is_ghost_type(self, pokemon):
        return pokemon.type_1 == "Ghost" or pokemon.type_2 == "Ghost"
    
    def fakeout_available(self, moves, opponent_pokemon):
        return any(move.id == "fakeout" and not self.is_ghost_type(opponent_pokemon) for move in moves)
    
    def choose_move(self, battle):
        if not battle.available_moves:
            return self.choose_random_move(battle)

        if battle.turn == 1 or self.last_pokemon != battle.active_pokemon.species:
            self.last_pokemon = battle.active_pokemon.species
            if self.fakeout_available(battle.available_moves, battle.opponent_active_pokemon):
                move = next(move for move in battle.available_moves if move.id == "fakeout")
                # print(f"Turn 1 or new active pokemon: {self.last_pokemon}")
                # print(f"Best Move: {move}")
                return self.create_order(move)

        if not self.protected and "protect" in [move.id for move in battle.available_moves]:
            best_move = max(battle.available_moves, key=lambda move: move.is_protect_move)
            self.protected = True
            return self.create_order(best_move)
        else:
            if battle.active_pokemon.current_hp_fraction < 0.55:
                best_move = max(battle.available_moves, key=lambda move: move.heal)
            else:
                best_move = max(battle.available_moves, key=lambda move: self.damage_multiplier(move, battle.opponent_active_pokemon))
                # print(f"Best Move: {best_move}")
            self.protected = False
            if "toxic" in [move.id for move in battle.available_moves] and battle.opponent_active_pokemon.status == None and battle.opponent_active_pokemon.type_1 != "Poison" and battle.opponent_active_pokemon.type_2 != "Poison" and battle.opponent_active_pokemon.type_1 != "Steel" and battle.opponent_active_pokemon.type_2 != "Steel":
                best_move = next(move for move in battle.available_moves if move.id == "toxic")
                return self.create_order(best_move)

        if battle.active_pokemon.current_hp_fraction < 0.45:
            if battle.active_pokemon.damage_multiplier(battle.opponent_active_pokemon.type_1) > 1 or \
                battle.active_pokemon.damage_multiplier(battle.opponent_active_pokemon.type_2) > 1:
                best_pokemon = max(battle.team.values(), key=lambda mon: (mon.current_hp_fraction, -mon.damage_multiplier(battle.opponent_active_pokemon.type_1), -mon.damage_multiplier(battle.opponent_active_pokemon.type_2)))
                return self.create_order(best_pokemon)

        self.last_pokemon = battle.active_pokemon.species
        return self.create_order(best_move)

    def damage_multiplier(self, move: Move, opponent: Pokemon):
        # Check if the move is a damaging move
        if move.base_power > 0:
            # Get the type multiplier
            type_multiplier = opponent.damage_multiplier(move.type)
            return move.base_power * type_multiplier
        else:
            return 0



# Main function
async def main():
    # We create the teams 
    # bot_team = await get_bot_team() 
    # enemy_team1 = await create_enemy_team()
    # enemy_team2 = await create_enemy_team()

    maxdamage_team = await get_maxdamage_team()
    maxdefense_team = await get_maxdefense_team()

    #We set the player configuration
    configuracion_max_damage=PlayerConfiguration("MaxDamage",None)
    configuracion_max_defense=PlayerConfiguration("MaxDefense",None)

    max_damage_player=MaxDamagePlayer(battle_format="gen7ou", team=maxdamage_team,player_configuration=configuracion_max_damage)
    max_defense_player=MaxDefensePlayer(battle_format="gen7ou", team=maxdefense_team,player_configuration=configuracion_max_defense)
    
    start = time.time()

    # await max_damage_player.battle_against(max_defense_player, n_battles=5000)

    await max_damage_player.send_challenges('DANIELmichel',1)

    print(f"total time of execution: {time.time() - start} seconds")
    print(f"From a total matches of {max_damage_player.n_won_battles + max_defense_player.n_won_battles} matches \nMaxDamagePlayer won {max_damage_player.n_won_battles} with a winrate of {max_damage_player.n_won_battles/(max_damage_player.n_won_battles + max_defense_player.n_won_battles)} \nMaxDefensePlayer won {max_defense_player.n_won_battles} with a winrate of {max_defense_player.n_won_battles/(max_damage_player.n_won_battles + max_defense_player.n_won_battles)}")

### Functions to create the teams

async def get_bot_team():
    with open('bot_team.json') as json_file:
        data = json.load(json_file)
        return data["team"] 
    
async def get_maxdamage_team():
    with open('maxdamage_team.json') as json_file:
        data = json.load(json_file)
        return data["team"] 

async def get_maxdefense_team():
    with open('maxdefense_team.json') as json_file:
        data = json.load(json_file)
        return data["team"] 


async def create_enemy_team():
    # Load the data from the JSON file
    with open('box.json', 'r') as json_file:
        data = json.load(json_file)
    
    # Get the list of available Pokémon
    pokemon_list = data['pokemons']
    
    # Randomly select six unique Pokémon
    enemy_team = random.sample(pokemon_list, k=6)
    
    # Format each Pokémon entry
    formatted_team = []
    for pokemon in enemy_team:
        formatted_pokemon = "\n".join(pokemon.split("\n")[:-1])  # Remove the last empty line
        formatted_team.append(formatted_pokemon)
    
    # Combine all the Pokémon entries into a single string
    enemy_team_formatted = "\n\n".join(formatted_team)
    
    return enemy_team_formatted


    
    
if __name__ == "__main__":
    asyncio.run(main())
