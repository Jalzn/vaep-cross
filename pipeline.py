import tracking.process
import tracking.read
from utils.events import generate_gamestates, read_actions

actions = read_actions()


tracking_df = tracking.read.read_by_match_id(13335)
tracking_df = tracking.process.process(tracking_df, actions, 13335)

gamestates = generate_gamestates(actions)


ids = gamestates[gamestates["match_id"] == 13335]["event_id"].tolist()

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

gamestates.head()