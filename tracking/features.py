from typing import Tuple
import pandas as pd
import numpy as np


def count_players_in_box(frame_df: pd.DataFrame, attacking_team_id: int) -> Tuple[int]:
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



def count_players_in_zone(frame: pd.DataFrame, action: pd.Series):
    def estimate_time_to_target(start_x, start_y, end_x, end_y, ball_speed=18):
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        time = distance / ball_speed
        return time

    def project_towards_target(player_x, player_y, target_x, target_y, time_delta, player_speed=1):
        # Calcula o vetor direção
        dx = target_x - player_x
        dy = target_y - player_y
        distance_to_target = np.sqrt(dx**2 + dy**2)
        
        if distance_to_target == 0:
            return player_x, player_y  # Já está no destino

        # Calcula o deslocamento em x e y
        direction_x = dx / distance_to_target
        direction_y = dy / distance_to_target
        
        # Quanto o jogador consegue percorrer no tempo disponível
        distance_covered = min(player_speed * time_delta, distance_to_target)

        new_x = player_x + direction_x * distance_covered
        new_y = player_y + direction_y * distance_covered

        return new_x, new_y

    def is_in_zone(player_x, player_y, target_x, target_y, radius=3):
        distance = np.sqrt((player_x - target_x)**2 + (player_y - target_y)**2)
        return distance <= radius

    num_attackers = 0
    num_defenders = 0

    team_id = action["team_id"]
    start_x = action["start_x"]
    start_y = action["start_y"]
    end_x = action["end_x"]
    end_y = action["end_y"]

    time_to_target = estimate_time_to_target(start_x, start_y, end_x, end_y)


    for _, player in frame.iterrows():
        projected_x, projected_y = project_towards_target(
            player_x=player['x'],
            player_y=player['y'],
            target_x=end_x,
            target_y=end_y,
            time_delta=time_to_target,
        )
        
        if is_in_zone(projected_x, projected_y, end_x, end_y, radius=3):
            if player['team_id'] == team_id:
                num_attackers += 1
            else:
                num_defenders += 1

    return num_attackers, num_defenders