import numpy as np

def prepare_pitch_control_input(frame_df, attacking_team_id, ball_x=0, ball_y=0):
    """
    Prepara os dados de um frame para o cálculo de Pitch Control.

    frame_df: DataFrame com todos os jogadores de um frame.
    attacking_team_id: int - ID do time que está atacando.
    ball_x, ball_y: posição da bola (ajuste se tiver tracking da bola real).

    Retorna:
        attacking_players: numpy array Nx4 [x, y, vx, vy]
        defending_players: numpy array Nx4 [x, y, vx, vy]
        ball_position: tuple (x, y)
    """
    attacking = frame_df[frame_df['team_id'] == attacking_team_id]
    defending = frame_df[frame_df['team_id'] != attacking_team_id]

    attacking_players = attacking[['x', 'y', 'vx', 'vy']].to_numpy()
    defending_players = defending[['x', 'y', 'vx', 'vy']].to_numpy()
    ball_position = (ball_x, ball_y)

    return attacking_players, defending_players, ball_position

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_time_to_intercept(player_pos, player_vel, target_pos, max_player_speed):
    """
    Calcula o tempo que um jogador leva para chegar até a bola.
    """
    reaction_time = 0.7  # segundos de atraso na reação
    distance_to_target = np.linalg.norm(target_pos - player_pos)
    relative_speed = max_player_speed + 1e-6  # evitar divisão por zero
    time_to_intercept = reaction_time + distance_to_target / relative_speed
    return time_to_intercept

def compute_pitch_control_at_target(target_pos, attacking_players, defending_players, params):
    """
    Calcula a probabilidade de controle do time atacante no ponto target_pos.
    """
    max_player_speed = params['max_player_speed']

    # Tempo de chegada dos atacantes e defensores
    t_attack = np.array([
        calculate_time_to_intercept(player[:2], player[2:], target_pos, max_player_speed)
        for player in attacking_players
    ])

    t_defense = np.array([
        calculate_time_to_intercept(player[:2], player[2:], target_pos, max_player_speed)
        for player in defending_players
    ])

    # Tempo mínimo de cada lado
    min_t_attack = np.min(t_attack) if len(t_attack) > 0 else np.inf
    min_t_defense = np.min(t_defense) if len(t_defense) > 0 else np.inf

    # Calcula probabilidade de controle
    lambda_att = 4.3  # parâmetro de transição
    delta_t = min_t_defense - min_t_attack

    p_attack = sigmoid(lambda_att * delta_t)

    return p_attack

def generate_pitch_control_for_frame(attacking_players, defending_players, ball_position, params=None):
    """
    Gera a matriz de Pitch Control para o campo todo.
    attacking_players e defending_players são arrays Nx4 com [x, y, vx, vy]
    ball_position é (x, y)
    """
    if params is None:
        params = {
            'max_player_speed': 5.0  # metros por segundo
        }

    # Define grid de pontos (exemplo: StatsBomb dimensions)
    x_grid = np.linspace(-60, 60, 100)  # ajuste conforme seu campo
    y_grid = np.linspace(-40, 40, 80)

    pitch_control_surface = np.zeros((len(y_grid), len(x_grid)))

    for ix, x in enumerate(x_grid):
        for iy, y in enumerate(y_grid):
            target_pos = np.array([x, y])
            p_attack = compute_pitch_control_at_target(
                target_pos,
                attacking_players,
                defending_players,
                params
            )
            pitch_control_surface[iy, ix] = p_attack

    return pitch_control_surface