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


def map_event_to_vaep_action(row):
    event_type = row['event_type'] if not pd.isna(row['event_type']) else "NA"
    pass_type = row['pass_type']if not pd.isna(row['pass_type']) else "NA"
    set_piece_type = row['set_piece_type'] if not pd.isna(row['set_piece_type']) else "NA"
    duel_type = row['duel_type'] if not pd.isna(row['duel_type']) else "NA"

    # --------- Passes e jogadas de bola parada -----------
    if event_type == 'PASS':
        if pass_type == 'CROSS':
            return 'cross'
        elif set_piece_type == 'CORNER_KICK':
            if pass_type == 'CROSS':
                return 'crossed_corner'
            else:
                return 'short_corner'
        elif set_piece_type == 'FREE_KICK':
            if pass_type == 'CROSS':
                return 'crossed_freekick'
            else:
                return 'short_freekick'
        elif set_piece_type == 'THROW_IN':
            return 'throw_in'
        else:
            return 'pass'

    # --------- Carregada (Carry / Dribble) -----------
    elif event_type == 'CARRY':
        return 'dribble'

    # --------- Take On -----------
    elif event_type == 'TAKE_ON':
        return 'take_on'

    # --------- Faltas -----------
    elif event_type == 'FOUL_COMMITTED':
        return 'foul'

    # --------- Duels (Tackle, Bad touch, Interception) -----------
    elif event_type == 'DUEL':
        if duel_type in ['SLIDING_TACKLE', 'GROUND']:
            return 'tackle'
        elif duel_type == 'LOOSE_BALL':
            return 'interception'
        elif duel_type == 'AERIAL':
            return 'tackle'  # ou outro, dependendo de como você quiser tratar

    # --------- Limpeza (Clearance) -----------
    elif event_type == 'CLEARANCE':
        return 'clearance'

    # --------- Finalizações -----------
    elif event_type == 'SHOT':
        if set_piece_type == 'PENALTY':
            return 'penalty_shot'
        elif set_piece_type == 'FREE_KICK':
            return 'freekick_shot'
        else:
            return 'shot'

    # --------- Goleiro -----------
    elif event_type == 'GOALKEEPER':
        # Aqui você pode ter campos adicionais para distinguir tipos de ações de goleiro
        # Exemplo genérico:
        return 'keeper_save'

    # --------- Outros -----------
    else:
        return 'non_action'  # Ou outro rótulo para eventos que você vai ignorar


def read_actions():
    actions = read_events()
    actions["action_type"] = actions.apply((lambda x: map_event_to_vaep_action(x).upper()), axis=1)

    return actions

def clean_actions(actions: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "match_id": actions["match_id"],
        "event_id": actions["event_id"],
        "team_id": actions["team_id"],
        "action_type": actions["action_type"],
        'start_x': actions['coordinates_x'] * 105 - 105/2,
        'start_y': actions['coordinates_y'] * 68 - 68/2,
        'end_x': actions['end_coordinates_x'].fillna(actions['coordinates_x']) * 105 - 105/2,
        'end_y': actions['end_coordinates_y'].fillna(actions['coordinates_y']) * 68 - 68/2,
    })

    return df

def calculate_labels(actions: pd.DataFrame) -> pd.DataFrame:
    actions['next_team_id'] = actions['team_id'].shift(-1)
    actions['cross_success'] = ((actions['action_type'] == 'CROSS') & (actions['team_id'] == actions['next_team_id'])).fillna(False).astype(int)
    actions.drop(columns=['next_team_id'], inplace=True)

    return actions

def standardize_cross_directions_top_down(df):
    # Se o cruzamento start_x atual for > 0, espelha todas as coordenadas no eixo X
    mask_flip = df['start_x'] < 0

    for i in ['2', '1', '']:
        sx = f'start_x_{i}' if i else 'start_x'
        ex = f'end_x_{i}' if i else 'end_x'
        df.loc[mask_flip, sx] = -df.loc[mask_flip, sx]
        df.loc[mask_flip, ex] = -df.loc[mask_flip, ex]

    # Padroniza eixo Y (de cima para baixo)
    mask_flip_y = df['start_y'] < 0
    for i in ['2', '1', '']:
        sy = f'start_y_{i}' if i else 'start_y'
        ey = f'end_y_{i}' if i else 'end_y'
        df.loc[mask_flip_y, sy] = -df.loc[mask_flip_y, sy]
        df.loc[mask_flip_y, ey] = -df.loc[mask_flip_y, ey]

    return df

def generate_gamestates(actions: pd.DataFrame) -> pd.DataFrame:
    actions = clean_actions(actions)
    actions = calculate_labels(actions)

    # Cria as colunas lag para ações e posições
    actions['action'] = actions['action_type']
    actions['action_1'] = actions['action_type'].shift(1)
    actions['action_2'] = actions['action_type'].shift(2)

    actions['start_x'] = actions['start_x']
    actions['start_x_1'] = actions['start_x'].shift(1)
    actions['start_x_2'] = actions['start_x'].shift(2)

    actions['start_y'] = actions['start_y']
    actions['start_y_1'] = actions['start_y'].shift(1)
    actions['start_y_2'] = actions['start_y'].shift(2)

    actions['end_x'] = actions['end_x']
    actions['end_x_1'] = actions['end_x'].shift(1)
    actions['end_x_2'] = actions['end_x'].shift(2)

    actions['end_y'] = actions['end_y']
    actions['end_y_1'] = actions['end_y'].shift(1)
    actions['end_y_2'] = actions['end_y'].shift(2)

    # Filtrar apenas os cruzamentos
    gamestates = actions[actions['action_type'] == 'CROSS'][[
        'match_id', 'event_id','team_id',
        'action_2', 'start_x_2', 'start_y_2', 'end_x_2', 'end_y_2',
        'action_1', 'start_x_1', 'start_y_1', 'end_x_1', 'end_y_1',
        'action', 'start_x', 'start_y', 'end_x', 'end_y', 'cross_success'
    ]].dropna().reset_index(drop=True)

    gamestates = standardize_cross_directions_top_down(gamestates)

    return gamestates