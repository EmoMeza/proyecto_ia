import asyncio
from poke_env.player_configuration import PlayerConfiguration

from poke_env.player import SimpleHeuristicsPlayer
from poke_env import ShowdownServerConfiguration



async def main():
    simple_heur = SimpleHeuristicsPlayer(battle_format="gen8randombattle", player_configuration=simple_heur_config, server_configuration=ShowdownServerConfiguration)

    await simple_heur.send_challenges('Dogepot', n_challenges=1)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())