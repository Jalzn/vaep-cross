from . import io, vaep

import pandas as pd

def convert_to_spadl(events_df):
    """Converte DataFrame de eventos para formato SPADL compatível com socceractions"""
    
    # Mapeamentos baseados no padrão SPADL
    action_type_mapping = {
        'PASS': {'id': 0, 'name': 'pass'},
        'DRIBBLE': {'id': 1, 'name': 'dribble'}, 
        'CARRY': {'id': 1, 'name': 'dribble'},
        'TAKE_ON': {'id': 1, 'name': 'dribble'},
        'SHOT': {'id': 2, 'name': 'shot'},
        'CROSS': {'id': 3, 'name': 'cross'},
        'THROW_IN': {'id': 4, 'name': 'throw_in'},
        'BALL_OUT': {'id': 4, 'name': 'throw_in'},
        'FREEKICK': {'id': 5, 'name': 'freekick'},
        'CORNER': {'id': 6, 'name': 'corner'},
        'TACKLE': {'id': 7, 'name': 'tackle'},
        'DUEL': {'id': 7, 'name': 'tackle'},
        'INTERCEPTION': {'id': 8, 'name': 'interception'},
        'CLEARANCE': {'id': 9, 'name': 'clearance'},
        'GOALKEEPER': {'id': 10, 'name': 'keeper_save'},
        'FOUL_COMMITTED': {'id': 11, 'name': 'foul'},
        'CARD': {'id': 12, 'name': 'card'},
        'SUBSTITUTION': {'id': 13, 'name': 'substitution'},
        'PLAYER_OFF': {'id': 14, 'name': 'player_off'},
        'PLAYER_ON': {'id': 15, 'name': 'player_on'},
        'OFFSIDE': {'id': 16, 'name': 'offside'}
    }
    
    result_mapping = {
        'COMPLETE': {'id': 1, 'name': 'success'},
        'WON': {'id': 1, 'name': 'success'},
        'GOAL': {'id': 1, 'name': 'success'},
        'INCOMPLETE': {'id': 0, 'name': 'fail'},
        'LOST': {'id': 0, 'name': 'fail'},
        'MISS': {'id': 0, 'name': 'fail'},
        'SAVED': {'id': 0, 'name': 'fail'},
        'BLOCK': {'id': 0, 'name': 'fail'}
    }
    
    bodypart_mapping = {
        'foot': {'id': 0, 'name': 'foot'},
        'foot_left': {'id': 1, 'name': 'foot_left'},
        'foot_right': {'id': 2, 'name': 'foot_right'},
        'head': {'id': 3, 'name': 'head'},
        'other': {'id': 4, 'name': 'other'},
        'chest': {'id': 4, 'name': 'other'},
        'shoulder': {'id': 4, 'name': 'other'},
        'knee': {'id': 4, 'name': 'other'}
    }
    
    # Função auxiliar para mapear valores
    def map_action_type(event_type):
        mapping = action_type_mapping.get(event_type, {'id': 17, 'name': 'other'})
        return pd.Series([mapping['id'], mapping['name']])
    
    def map_result(result_val):
        if pd.isna(result_val):
            return pd.Series([0, 'fail'])
        mapping = result_mapping.get(result_val, {'id': 0, 'name': 'fail'})
        return pd.Series([mapping['id'], mapping['name']])
    
    def map_bodypart(bodypart_val):
        if pd.isna(bodypart_val):
            bodypart_val = 'foot'
        bodypart_val = str(bodypart_val).lower()
        mapping = bodypart_mapping.get(bodypart_val, {'id': 0, 'name': 'foot'})
        return pd.Series([mapping['id'], mapping['name']])
    
    # Criar DataFrame SPADL
    spadl_df = pd.DataFrame()
    
    # Campos básicos
    spadl_df['game_id'] = events_df['match_id']
    spadl_df['original_event_id'] = events_df.get('event_id', range(len(events_df)))
    spadl_df['period_id'] = events_df['period_id']
    
    # Timestamp - converter para segundos
    if 'timestamp' in events_df.columns:
        if hasattr(events_df['timestamp'].iloc[0] if len(events_df) > 0 else None, 'total_seconds'):
            spadl_df['time_seconds'] = events_df['timestamp'].apply(lambda x: x.total_seconds() if pd.notna(x) else 0)
        else:
            spadl_df['time_seconds'] = pd.to_numeric(events_df['timestamp'], errors='coerce').fillna(0)
    else:
        spadl_df['time_seconds'] = 0
    
    # IDs de time e jogador
    spadl_df['team_id'] = events_df['team_id']
    spadl_df['player_id'] = events_df['player_id']
    
    # Coordenadas - converter para campo de 105x68
    spadl_df['start_x'] = events_df['coordinates_x'] * 105
    spadl_df['start_y'] = events_df['coordinates_y'] * 68
    spadl_df['end_x'] = events_df['end_coordinates_x'].fillna(events_df['coordinates_x']) * 105
    spadl_df['end_y'] = events_df['end_coordinates_y'].fillna(events_df['coordinates_y']) * 68
    
    # Mapear tipos de ação
    action_mapping = events_df['event_type'].apply(lambda x: map_action_type(x))
    spadl_df['type_id'] = action_mapping.iloc[:, 0]
    spadl_df['type_name'] = action_mapping.iloc[:, 1]
    
    # Mapear resultados
    if 'result' in events_df.columns and events_df['result'].notna().any():
        result_mapping_series = events_df['result'].apply(lambda x: map_result(x))
    else:
        # Usar coluna success se disponível
        success_col = events_df.get('success', False)
        result_mapping_series = success_col.apply(lambda x: map_result('COMPLETE' if x else 'INCOMPLETE'))
    
    spadl_df['result_id'] = result_mapping_series.iloc[:, 0]
    spadl_df['result_name'] = result_mapping_series.iloc[:, 1]
    
    # Mapear partes do corpo
    bodypart_col = events_df.get('body_part_type', 'foot')
    bodypart_mapping_series = bodypart_col.apply(lambda x: map_bodypart(x))
    spadl_df['bodypart_id'] = bodypart_mapping_series.iloc[:, 0]
    spadl_df['bodypart_name'] = bodypart_mapping_series.iloc[:, 1]
    
    # action_id sequencial
    spadl_df['action_id'] = range(len(spadl_df))
    
    # Reordenar colunas conforme padrão SPADL
    column_order = [
        'game_id', 'original_event_id', 'period_id', 'time_seconds', 
        'team_id', 'player_id', 'start_x', 'start_y', 'end_x', 'end_y',
        'type_id', 'result_id', 'bodypart_id', 'action_id',
        'type_name', 'result_name', 'bodypart_name'
    ]
    
    # Manter apenas colunas que existem
    existing_columns = [col for col in column_order if col in spadl_df.columns]
    spadl_df = spadl_df[existing_columns]
    
    return spadl_df
    