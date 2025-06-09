import gandula
from gandula.export.dataframe import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed

import pandas as pd
import numpy as np


def smooth_velocity_by_player(df, window=5):
    """
    Suaviza as colunas 'vx', 'vy' e 'speed' usando média móvel,
    agrupando por jogador identificado pela coluna 'shirt'.

    Parâmetros:
        df (pd.DataFrame): DataFrame com as colunas 'vx', 'vy', 'speed' e 'shirt'.
        window (int): tamanho da janela para a média móvel.

    Retorna:
        pd.DataFrame com as colunas suavizadas.
    """
    df = df.copy()
    df[['vx_smooth', 'vy_smooth', 'speed_smooth']] = (
        df.groupby('shirt')[['vx', 'vy', 'speed']]
          .transform(lambda x: x.rolling(window=window, min_periods=1, center=True).mean())
    )
    return df


def standardize_attack_direction(df_frames):
    """
    Padroniza a direção de ataque dos cruzamentos para que todos ocorram da esquerda para a direita

    Parâmetros:
        df_frames (pd.DataFrame): DataFrame contendo os dados de tracking dos jogadores e da bola, já
                                  segmentados por janelas de cruzamentos. Espera-se que contenha as colunas:
                                  ['x', 'vx', 'ax', 'ball_x', 'ball_vx', 'ball_ax', 'vx_smooth', 'cross_id',
                                  'possession_type'].

    Retorna:
        pd.DataFrame: Uma cópia do DataFrame original, com todos os cruzamentos padronizados para
                      ocorrerem no sentido da esquerda para a direita (x positivo).
    """

    df = df_frames.copy()
    mask_cross = df["possession_type"] == "Frame_PossessionEventType.CROSS"
    cross_ids = df.loc[mask_cross, "cross_id"].dropna().unique()

    if len(cross_ids) == 0:
        return df

    ball_x_cross = (
        df[mask_cross]
        .groupby("cross_id")["ball_x"]
        .mean()
        .to_dict()
    )

    # Identificar os cross_id onde a bola estava do lado esquerdo
    cross_to_flip = {
        cross_id for cross_id, ball_x in ball_x_cross.items()
        if pd.notna(ball_x) and ball_x < 0
    }
    mask_to_flip = df["cross_id"].isin(cross_to_flip)

    cols_to_flip = [
        "x", "vx", "ball_x",
    ]

    # Aplicar espelhamento
    df.loc[mask_to_flip, cols_to_flip] = -df.loc[mask_to_flip, cols_to_flip]

    return df


def _process_metadata_pff(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa os dados de metadata da PFF, limpando campos inuteis,
    trasnformando colunas para tipo correto e etc 
    """
    df = metadata_df.copy()

    df = df.drop(["match_id", "home_has_possession"], axis=1)

    to_int_columns = [
        "frame_id", "event_id", "event_setpiece_type", "event_player_id",
        "event_team_id", "event_start_frame", "event_end_frame", "possession_id",
        "possession_start_frame", "possession_end_frame",
    ]

    df[to_int_columns] = df[to_int_columns].astype('Int64')

    df.replace({"possession_type": "nan"}, np.nan, inplace=True)
    df.replace({"event_type": "nan"}, np.nan, inplace=True)

    df = df[df["possession_type"].notna()].reset_index(drop=True)

    df = df.drop(["possession_end_frame", "sequence", "version",
                 "video_time_milli", "event_setpiece_type"], axis=1)

    return df


def _process_players_pff(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa os dados de jogadores da PFF, limpando campos inutueis,
    transformando colunas para tipo correto e etc. Alem disso, cria
    os campos de velocidade suavizada e espelha o campo para o lado
    correto

    Atencao: Essa funcao so pode ser chamada apos o process_metadata_pff
    e o merge entre os dois
    """
    players_df["frame_id"] = players_df["frame_id"].astype(int)

    players_df = smooth_velocity_by_player(players_df)

    players_df = players_df.sort_values(
        by=["frame_id", "team", "shirt"]).reset_index(drop=True)

    players_df = players_df.drop(["vx", "vy", "speed"], axis=1)

    players_df["vx"] = players_df["vx_smooth"]
    players_df["vy"] = players_df["vy_smooth"]
    players_df["speed"] = players_df["speed_smooth"]

    players_df = players_df.drop(
        ["vx_smooth", "vy_smooth", "speed_smooth"], axis=1)

    players_df = standardize_attack_direction(players_df)

    return players_df


def create_match_cross_df(players_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma dataframe que e o merge dos players com metadados alem de aplicar a
    padronizacao de ataque para apenas cruzamentos em uma janela de 3 seg, alem
    de ajustar a direcao de ataques
    """
    metadata_df = _process_metadata_pff(metadata_df)

    players_df["frame_id"] = players_df["frame_id"].astype(int)

    ids = []

    for i in metadata_df[metadata_df["possession_type"] == "Frame_PossessionEventType.CROSS"]["frame_id"].unique():
        for j in range(i - 120, i + 121):
            ids.append(j)

    players_df = players_df[players_df["frame_id"].isin(ids)]

    df_cross = pd.merge(
        players_df[["frame_id", "period", "shirt", "x", "y", "team",
                    "ball_x", "ball_y", "ball_z", "vx", "vy", "speed"]],
        metadata_df[["frame_id", "possession_type"]],
        on="frame_id",
        how="left"
    )

    cross_id = 0

    for i in df_cross[df_cross["possession_type"] == "Frame_PossessionEventType.CROSS"]["frame_id"].unique():
        mask = df_cross["frame_id"].isin(range(i - 120, i + 121))
        df_cross.loc[mask, "cross_id"] = cross_id
        cross_id += 1

    df_cross = _process_players_pff(df_cross)

    return df_cross


def preprocess_pff(player_df, metadata_df):
    """
    Pre processa os dados da pff   
    """
    metadata_df = metadata_df.copy()
    player_df = player_df.copy()

    ##### Pre processamento dos dados #####
    for col in metadata_df.select_dtypes(include=['object']).columns:
        metadata_df[col] = metadata_df[col].astype(str)

    for col in players_df.select_dtypes(include=['object']).columns:
        players_df[col] = players_df[col].astype(str)

    players_df = add_ball_speed(players_df)
    players_df = add_players_speed(players_df)

    # Coloca os frame_id como inteiros
    players_df["frame_id"] = players_df["frame_id"].astype(int)
    metadata_df["frame_id"] = metadata_df["frame_id"].astype(int)


def process_pff(data_path: str, game_id: int):
    """
    Processa um arquivo bz2 de dados da pff e salva os dados de tracking
    dos jogadores e os metadados

    Args:
        data_path: O caminho para o diretorio com os bz2
        game_id: O id do jogo
    """
    metadata_df, players_df = pff_frames_to_dataframe(
        gandula.get_frames(
            data_path,
            game_id,
        )
    )

    ##### pre processamento dos dados #####
    for col in metadata_df.select_dtypes(include=['object']).columns:
        metadata_df[col] = metadata_df[col].astype(str)

    for col in players_df.select_dtypes(include=['object']).columns:
        players_df[col] = players_df[col].astype(str)

    players_df = add_ball_speed(players_df)
    players_df = add_players_speed(players_df)

    # Coloca os frame_id como inteiros
    players_df["frame_id"] = players_df["frame_id"].astype(int)
    metadata_df["frame_id"] = metadata_df["frame_id"].astype(int)

    # Trata os nans em formato de string para o formato correto
    metadata_df.replace({"possession_type": "nan"}, np.nan, inplace=True)
    metadata_df.replace({"event_type": "nan"}, np.nan, inplace=True)

    metadata_df.to_parquet("./metadata.parquet", engine="fastparquet")
    players_df.to_parquet("./players.parquet", engine="fastparquet")
