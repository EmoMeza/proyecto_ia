import json

# Read the contents of the file
with open('box.txt', 'r') as file:
    file_contents = file.read()

# Split the contents into individual Pokémon entries
pokemon_entries = file_contents.split('\n\n')

# Create a dictionary with the "pokemons" key and the Pokémon entries as values
data = {
    "pokemons": pokemon_entries
}

# Write the data dictionary to the JSON file
with open('box.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
