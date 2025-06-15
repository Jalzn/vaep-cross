import os

import gandula
from gandula.export.dataframe import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed

import pff

bz2_files = [f for f in os.listdir("./data/") if f.endswith(".bz2")]

data_path, game_id = "./data", 13621

metadata_df, players_df = pff_frames_to_dataframe(
    gandula.get_frames(
        data_path,
        game_id,
    )
)

for col in metadata_df.select_dtypes(include=['object']).columns:
    metadata_df[col] = metadata_df[col].astype(str)

for col in players_df.select_dtypes(include=['object']).columns:
    players_df[col] = players_df[col].astype(str)


players_df = add_ball_speed(players_df)
players_df = add_players_speed(players_df)

df_cross = pff.create_match_cross_df(players_df, metadata_df)

df_cross.to_parquet(f"./data/cross_{game_id}.parquet", engine="fastparquet")
