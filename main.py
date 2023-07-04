import asyncio
import time
import numpy as np
import json
import random
import poke_env
from poke_env.player import Player, RandomPlayer
from poke_env.player_configuration import PlayerConfiguration



#Code to create a player that always chooses the move with the highest base power
class MaxDamagePlayer(Player): #borrar esto y pegar lo que esta en el main sobre max damage player
    def choose_move(self, battle):
        last_pokemon = None
        if battle.available_moves:

            if battle.turn == 1:
                if last_pokemon != battle.active_pokemon:
                    for move in battle.available_moves:
                        if move.id == "fakeout" and battle.opponent_active_pokemon.type_1 != "Ghost" and battle.opponent_active_pokemon.type_2 != "Ghost":
                            last_pokemon = battle.active_pokemon
                            return self.create_order(move)
                
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
            last_pokemon = battle.active_pokemon
        else:
            return self.choose_random_move(battle)
            last_pokemon = battle.active_pokemon
        
class MaxDefensePlayer(Player):
    last_pokemon = ""
    protected = False
    def choose_move(self, battle):
        if battle.available_moves:
            print(battle.active_pokemon.species) #Revisar como se llama el pokemon activo y guardarlo en una variable para checkear si es el mismo
            if battle.turn == 1:
                if last_pokemon != battle.active_pokemon.data["pokedex"]:#Revisar como se llama el pokemon activo y guardarlo en una variable para checkear si es el mismo
                    for move in battle.available_moves:
                        if move.id == "fakeout" and battle.opponent_active_pokemon.type_1 != "Ghost" and battle.opponent_active_pokemon.type_2 != "Ghost":
                            last_pokemon = battle.active_pokemon.data["pokedex"]#Revisar como se llama el pokemon activo y guardarlo en una variable para checkear si es el mismo
                            return self.create_order(move)
            
                
            if self.protected == False:
                best_move = max(battle.available_moves, key=lambda move: move.is_protect_move)
                self.protected = True
            else:
                if battle.active_pokemon.current_hp_fraction < 0.45:
                    best_move = max(battle.available_moves, key=lambda move: move.heal)
                else:
                    best_move = max(battle.available_moves, key=lambda move: move.base_power)
                self.protected = False
            last_pokemon = battle.active_pokemon.data["pokedex"]
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


# Main function
async def main():
    # We create the teams 
    bot_team = await get_bot_team() 
    enemy_team = await create_enemy_team()

    #We set the player configuration
    uc=PlayerConfiguration("Equipobot",None)
    mdp = MaxDefensePlayer(battle_format="gen7ou", team=bot_team,player_configuration=uc)

    # We create a player that will play randomly as an opponent
    ramdom_player = RandomPlayer(battle_format="gen7ou", save_replays=True, team= enemy_team)


    # The following are the setup for the Code to run
    # number_of_teams = input("How many teams do you want to battle against? ")
    # number_of_teams = int(number_of_teams)

    matches_per_team = input("How many matches do you want to play per team? ")
    matches_per_team = int(matches_per_team)

    # total_matches = number_of_teams * matches_per_team

    # Start the timer
    start = time.time()

    # Loop where the battles are played and the teams are updated
    # for x in range(number_of_teams):
    await mdp.battle_against(ramdom_player, n_battles=matches_per_team)
    enemy_team = await create_enemy_team()
    
    #update_team is a method that I (Emo) created in the Player class to update the team
    # await ramdom_player.update_team(enemy_team)

    # Final results
    # print(f"From a total of matches, the bot won {max_damage_player.n_won_battles} matches.\nGetting a win rate of {max_damage_player.n_won_battles/total_matches*100}% \nTime elapsed: {time.time() - start} seconds")


### Functions to create the teams

async def get_bot_team():
    with open('bot_team.json') as json_file:
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
