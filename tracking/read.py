import ast
import pandas as pd

def read_by_match_id(match_id: int) -> pd.DataFrame:
    tracking_df = pd.read_parquet(f"./data/{match_id}.parquet")

    metadata_df = pd.read_csv("./data/metadata.csv")

    metadata_df = metadata_df[metadata_df["id"] == match_id]

    metadata_df["homeTeam"] = metadata_df["homeTeam"].apply(ast.literal_eval)
    metadata_df["awayTeam"] = metadata_df["awayTeam"].apply(ast.literal_eval)

    homeTeam = pd.to_numeric(metadata_df.iloc[0]["homeTeam"]["id"])
    awayTeam = pd.to_numeric(metadata_df.iloc[0]["awayTeam"]["id"])

    team_lookup = {'home': homeTeam, 'away': awayTeam}  # Ajuste conforme o jogo

    tracking_df['team_id'] = tracking_df['element'].map(team_lookup)

    tracking_df = tracking_df.rename(columns={
        "gameRefId": "match_id",
        "periodGameClockTime": "period_game_clock",
        "frameNum": "frame_num",
        "jerseyNum": "jersey_number"
    })

    tracking_df["match_id"] =  pd.to_numeric(tracking_df["match_id"])
    tracking_df["frame_num"] =  pd.to_numeric(tracking_df["frame_num"])
    tracking_df["jersey_number"] =  pd.to_numeric(tracking_df["jersey_number"])

    tracking_df = tracking_df[tracking_df['element'] != 'ball'].reset_index(drop=True)

    return tracking_df 