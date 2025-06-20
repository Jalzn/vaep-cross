import pandas as pd

def _calculate_smoothed_velocity(tracking_df: pd.DataFrame, window_size=8, frame_rate=25) -> pd.DataFrame:
    """
    Calcula a velocidade suavizada (vx, vy) dos jogadores a partir do tracking_df.

    Parâmetros:
    - tracking_df: DataFrame com colunas ['frame_num', 'jersey_number', 'x', 'y']
    - window_size: Tamanho da janela de suavização (número de frames)
    - frame_rate: Frames por segundo (ex: 25 fps padrão em muitos datasets)

    Retorna:
    - DataFrame original com colunas novas: ['vx', 'vy']
    """
    
    df = tracking_df.copy()
    
    # Ordena para garantir que os dados estejam por jogador e por tempo
    df = df.sort_values(by=['jersey_number', 'frame_num']).reset_index(drop=True)
    
    # Função auxiliar: calcular derivada numérica central (posição / tempo)
    def calc_velocity(group):
        dt = 1 / frame_rate
        group['vx'] = group['x'].diff().rolling(window=window_size, center=True, min_periods=1).mean() / dt
        group['vy'] = group['y'].diff().rolling(window=window_size, center=True, min_periods=1).mean() / dt
        return group

    # Aplica por jogador
    df = df.groupby('jersey_number', group_keys=False).apply(calc_velocity)

    return df


def _standardize_crossings_direction(cross_tracking_df, cross_events_df):
    df = cross_tracking_df.copy()

    for _, event in cross_events_df.iterrows():
        event_id = event["event_id"]
        team_id = event["team_id"]
        player_jersey_num = event["player_jersey_num"]


        mask = df["possession_event_id"] == event_id
        frame = df.loc[mask]

        player_with_ball = frame[
            (frame["jersey_number"] == player_jersey_num) &
            (frame["team_id"] == team_id)
        ]

        if player_with_ball.empty:
            continue

        player_with_ball = frame[
            (frame["jersey_number"] == player_jersey_num) &
            (frame["team_id"] == team_id)
        ].iloc[0]

        if player_with_ball["x"] < 0:
            df.loc[mask, "x"] = df.loc[mask, "x"] * -1
            df.loc[mask, "vx"] = df.loc[mask, "vx"] * -1

        if player_with_ball["y"] < 0:
            df.loc[mask, "y"] = df.loc[mask, "y"] * -1
            df.loc[mask, "vy"] = df.loc[mask, "vy"] * -1

    return df

def process(tracking_df: pd.DataFrame, actions: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    Processa os dados brutos de tracking vindo do parquet.
    Vai adicionar as velocidades suavizadas, filtrar apenas os
    frames de cruzamento e padronizar para tudo ocorrer no mesmo
    lado do campo
    """
    tracking_df = _calculate_smoothed_velocity(tracking_df)

    actions = actions[
        (actions["action_type"] == "CROSS") &
        (actions["match_id"] == match_id)
    ].copy()
    
    tracking_df = tracking_df[tracking_df["possession_event_id"].isin(actions["event_id"].tolist())].copy()

    tracking_df = _standardize_crossings_direction(tracking_df, actions)

    return tracking_df