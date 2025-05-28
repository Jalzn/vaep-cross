import os

import gandula
from gandula.export.dataframe import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed

bz2_files = [f for f in os.listdir("./data/") if f.endswith(".bz2")]

data_path, game_id = "./data", 13621

metadata_df, players_df = pff_frames_to_dataframe(
    gandula.get_frames(
        data_path,
        game_id,
    )
)

players_df = add_ball_speed(players_df)
players_df = add_players_speed(players_df)

metadata_df.to_parquet("./metadata.parquet", engine="fastparquet")
players_df.to_parquet("./players.parquet", engine="fastparquet")
