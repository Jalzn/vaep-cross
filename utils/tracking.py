import pandas as pd


def calculate_smoothed_velocity(tracking_df: pd.DataFrame, window_size=8, frame_rate=25) -> pd.DataFrame:
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

def standardize_crossings_direction(cross_tracking_df, cross_events_df):
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
            df.loc[mask, "y"] = df.loc[mask, "y"] * -1
            df.loc[mask, "vy"] = df.loc[mask, "vy"] * -1

    return df


def mark_cross_success(events_df):
    events = events_df.copy()
    events["cross_success"] = False  # Inicializa a coluna

    # Garante ordenação temporal
    events = events.sort_values(["match_id", "period_id", "timestamp"]).reset_index(drop=True)

    cross_indices = events[events["pass_type"] == "CROSS"].index

    for idx in cross_indices:
        if idx + 1 >= len(events):
            continue  # Não tem próximo evento

        next_event = events.iloc[idx + 1]

        # Exemplo: se o próximo evento for do mesmo time e tiver resultado "COMPLETE"
        if  pd.notna(next_event["result"]) and next_event["result"] == "COMPLETE":
            events.at[idx, "cross_success"] = True

    return events


def count_players_in_box(frame_df, attacking_team_id):
    """
    Conta número de atacantes e defensores dentro da área para um frame específico.

    Parâmetros:
        frame_df: DataFrame com tracking de um único frame (posição x, y, team_id de cada jogador).
        attacking_team_id: ID do time que está atacando.

    Retorno:
        attackers_in_box: Número de atacantes na área
        defenders_in_box: Número de defensores na área
    """

    # Define limites da área (ajuste se seu campo tiver outra escala)
    area_x_min = 36.5   # Começo da grande área no lado direito (assumindo ataque da esquerda pra direita)
    area_x_max = 60     # Fim do campo
    area_y_min = -20
    area_y_max = 20

    # Filtra jogadores dentro da área
    in_box = frame_df[
        (frame_df["x"] >= area_x_min) &
        (frame_df["x"] <= area_x_max) &
        (frame_df["y"] >= area_y_min) &
        (frame_df["y"] <= area_y_max)
    ]

    # Conta
    attackers_in_box = (in_box["team_id"] == attacking_team_id).sum()
    defenders_in_box = (in_box["team_id"] != attacking_team_id).sum()

    return attackers_in_box, defenders_in_box
