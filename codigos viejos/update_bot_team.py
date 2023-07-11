import json

def read_team_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()

def create_json_file(file_name, team_data):
    team_json = {
        "team": team_data.strip()
    }
    with open(file_name, 'w') as file:
        json.dump(team_json, file, indent=4)

# Name of the input text file
input_file = "bot_team.txt"
# Name of the output JSON file
output_file = "bot_team.json"

# Read the team data from the input file
team_data = read_team_file(input_file)

# Create the JSON file with the team data
create_json_file(output_file, team_data)

print("team.json file created successfully.")
