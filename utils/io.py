import ast
import pandas as pd

def _read_roster() -> pd.DataFrame:
    roster_df = pd.read_csv("./data/rosters.csv")

    roster_df['player'] = roster_df['player'].apply(ast.literal_eval)
    roster_df['team'] = roster_df['team'].apply(ast.literal_eval)

    roster_df['match_id'] = pd.to_numeric(roster_df['game_id'])

    # Extrair player_id
    roster_df['player_id'] = pd.to_numeric(roster_df['player'].apply(lambda x: x['id']))

    # Extrair team_id e team_name
    roster_df['team_id'] = pd.to_numeric(roster_df['team'].apply(lambda x: x['id']))
    roster_df['team_name'] = roster_df['team'].apply(lambda x: x['name'])

    # Extrair nickname (se quiser)
    roster_df['player_nickname'] = roster_df['player'].apply(lambda x: x['nickname'])

    # Selecionar colunas finais no estilo do roster
    roster_df = roster_df.rename(columns={
        'positionGroupType': 'position_group_type',
        'shirtNumber': 'jersey_number'
    })[[
        'match_id',
        'player_id',
        'player_nickname',
        'position_group_type',
        'jersey_number',
        'team_id',
        'team_name'
    ]]


    return roster_df

def read_events() -> pd.DataFrame:
    roster_df = _read_roster()
    events_df = pd.read_parquet("./data/eventos_sem_generic.parquet")

    int_columns = ['event_id', 'period_id', 'match_id', 'player_id', 'team_id', 'receiver_player_id']
    float_columns = ['coordinates_x', 'coordinates_y', 'end_coordinates_x', 'end_coordinates_y']

    # Converter colunas inteiras
    for col in int_columns:
        events_df[col] = pd.to_numeric(events_df[col], errors='coerce').astype('Int64')

    # Converter colunas float
    for col in float_columns:
        events_df[col] = pd.to_numeric(events_df[col], errors='coerce')

    # Converter campos de timestamp (se quiser como datetime)
    # events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], errors='coerce')
    # events_df['end_timestamp'] = pd.to_datetime(events_df['end_timestamp'], errors='coerce')

    # Campos de categorias textuais (como event_type, result, pass_type, etc)
    categorical_columns = [
        'event_type', 'ball_state', 'ball_owning_team', 'body_part_type',
        'set_piece_type', 'pass_type', 'result', 'duel_type',
        'goalkeeper_type', 'card_type'
    ]

    for col in categorical_columns:
        events_df[col] = events_df[col].astype('string').fillna(pd.NA)
    
    # Campo booleano (success), garantir coerência
    events_df['success'] = events_df['success'].map({'True': True, 'False': False}).astype('boolean')

    roster_lookup = roster_df[['match_id', 'player_id', 'jersey_number', 'position_group_type']]

    events_df = events_df.merge(
        roster_lookup.rename(columns={
            'jersey_number': 'player_jersey_num',
            'position_group_type': 'player_position_group_type'
        }),
        on=['match_id', 'player_id'],
        how='left'
    )

    events_df = events_df.merge(
        roster_lookup.rename(columns={
            'player_id': 'receiver_player_id',
            'jersey_number': 'receiver_jersey_num',
            'position_group_type': 'receiver_position_group_type'
        }),
        on=['match_id', 'receiver_player_id'],
        how='left'
    )

    return events_df

def read_tracking(match_id: int) -> pd.DataFrame:
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

    # roster_df = _read_roster()

    # tracking_df = tracking_df.merge(
    #     roster_df[['team_id', 'jersey_number', 'player_id']],
    #     on=['team_id', 'jersey_number'],
    #     how='left'
    # )

    return tracking_df 