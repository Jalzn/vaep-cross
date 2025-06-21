import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import tracking.features
import tracking.process
import tracking.read
from utils.events import generate_gamestates, read_actions



if not (Path("./data/gamestates.parquet").exists() and Path("./data/actions.parquet").exists()):
    print("Creating raw actions and gamestates")
    actions = read_actions()
    gamestates = generate_gamestates(actions)
    actions.to_parquet("./data/actions.parquet")
    gamestates.to_parquet("./data/gamestates.parquet")
else:
    print("Loading actions and gamestates")
    actions = pd.read_parquet("./data/actions.parquet")
    gamestates = pd.read_parquet("./data/gamestates.parquet")


tracking_folder = Path("./data/")
parquet_pattern = re.compile(r'^\d{5}\.parquet$')

files = [f for f in tracking_folder.iterdir() if f.is_file() and parquet_pattern.match(f.name)]

matches = []
for file in files:
    pattern = re.match(r'^(\d{5})\.parquet$', file.name)
    if pattern:
        matches.append(int(pattern.group(1)))

print(f"Found {len(matches)} matches")


for match_id in tqdm(matches):
    tracking_df = tracking.read.read_by_match_id(match_id)
    tracking_df = tracking.process.process(tracking_df, actions, match_id)

    ids = gamestates[gamestates["match_id"] == match_id]["event_id"].tolist()

    for id in ids:
        frame = tracking_df[tracking_df["possession_event_id"] == id]

        event = gamestates[gamestates["event_id"] == id].iloc[0]
        home_team_id = event["team_id"]

        attackers, defenders = tracking.features.count_players_in_box(frame, home_team_id)

        gamestates.loc[gamestates["event_id"] == id, "attackers_in_box" ] = attackers
        gamestates.loc[gamestates["event_id"] == id, "defenders_in_box"] = defenders

    for id in ids:
        frame = tracking_df[tracking_df["possession_event_id"] == id]

        event = gamestates[gamestates["event_id"] == id].iloc[0]

        attackers, defenders = tracking.features.count_players_in_zone(frame, event)

        gamestates.loc[gamestates["event_id"] == id, "attackers_in_zone" ] = attackers
        gamestates.loc[gamestates["event_id"] == id, "defenders_in_zone"] = defenders

    gamestates.to_parquet("./data/gamestates.parquet")

print("Finished")