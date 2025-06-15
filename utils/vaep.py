
import datetime
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from socceraction.vaep import VAEP
import joblib
from tqdm import tqdm

import socceraction.vaep as vaep
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab

def prepare_vaep_data(actions):
    """
    Prepara os dados para treinamento do VAEP
    
    Args:
        actions (pd.DataFrame): DataFrame com dados SPADL
    
    Returns:
        tuple: (gamestates, labels_scores, labels_concedes)
    """
    home_team_id = actions.iloc[0]["home_team_id"]

    print("Preparando dados para VAEP...")
    print("Convertendo ações para game states...")
    gamestates = fs.gamestates(actions, nb_prev_actions=3)
    gamestates = fs.play_left_to_right(gamestates, home_team_id)
    print(f"Game states criados: {len(gamestates)}")
    
    print("Extraindo features dos game states...")
    xfns = [
        fs.actiontype_onehot,
        fs.bodypart_onehot,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        fs.time_delta
    ]
    X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
    print(f"Features extraídas: {X.shape}")
    
    print("Criando labels...")
    yfns = [lab.scores, lab.concedes]
    Y = pd.concat([fn(actions) for fn in yfns], axis=1)
    
    print(f"Labels antes da limpeza:")
    print(f"  Scoring: {len(Y['scores'])} total, {Y['scores'].isna().sum()} NA values")
    print(f"  Conceding: {len(Y['concedes'])} total, {Y['concedes'].isna().sum()} NA values")
    
    # 4. Tratar valores pd.NA nos labels
    # Alinhar X e Y (gamestates vs actions podem ter tamanhos diferentes)
    min_length = min(len(X), len(Y))
    X = X.iloc[:min_length].copy()
    Y = Y.iloc[:min_length].copy()
    
    # Identificar linhas com valores válidos (não NA)
    valid_mask = ~(Y['scores'].isna() | Y['concedes'].isna())
    
    print(f"Removendo {(~valid_mask).sum()} linhas com valores NA")
    print(f"Dados válidos restantes: {valid_mask.sum()}")
    
    # Filtrar apenas dados válidos
    X_clean = X[valid_mask].copy()
    Y_clean = Y[valid_mask].copy()
    # gamestates_clean = gamestates.iloc[:min_length][valid_mask].copy()
    
    # Verificar se ainda temos dados suficientes
    if len(X_clean) == 0:
        raise ValueError("Nenhum dado válido após remoção de valores NA")
    
    if len(X_clean) < 100:
        print(f"AVISO: Apenas {len(X_clean)} amostras válidas. Considere usar mais dados.")
    
    print(f"Dados finais:")
    print(f"  Features: {X_clean.shape}")
    print(f"  Labels scoring: {len(Y_clean['scores'])} (positivos: {Y_clean['scores'].sum()})")
    print(f"  Labels conceding: {len(Y_clean['concedes'])} (positivos: {Y_clean['concedes'].sum()})")
    
    return X_clean, Y_clean



def train_vaep_model(X, Y, test_size=0.2, random_state=42):
    """
    Treina o modelo VAEP
    
    Args:
        X (pd.DataFrame): Features dos game states
        Y (pd.DataFrame): Labels com colunas 'scores' e 'concedes'
        test_size (float): Proporção dos dados para teste
        random_state (int): Seed para reprodutibilidade
    
    Returns:
        tuple: (model_score, model_concede) Modelos treinados
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report
    
    print("Dividindo dados em treino e teste...")
    
    # Extrair labels separadamente
    y_scores = Y['scores'].astype(int)
    y_concedes = Y['concedes'].astype(int)
    
    # Dividir dados em treino e teste
    X_train, X_test, y_score_train, y_score_test, y_concede_train, y_concede_test = train_test_split(
        X, y_scores, y_concedes, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_scores  # Estratificar baseado em scoring para manter proporção
    )
    
    print(f"Dados de treino: {len(X_train)}")
    print(f"Dados de teste: {len(X_test)}")
    
    # Criar e treinar modelos separados para scoring e conceding
    print("Iniciando treinamento dos modelos...")
    
    # Modelo para scoring
    print("Treinando classificador para scoring...")
    model_score = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    model_score.fit(X_train, y_score_train)
    
    # Modelo para conceding
    print("Treinando classificador para conceding...")
    model_concede = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    model_concede.fit(X_train, y_concede_train)
    
    print("Modelos treinados com sucesso!")
    
    # Avaliar modelos
    print("Avaliando modelos...")
    
    # Predições para scoring
    score_predictions = model_score.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva
    try:
        score_auc = roc_auc_score(y_score_test, score_predictions)
        print(f"AUC Score (Scoring): {score_auc:.4f}")
    except ValueError as e:
        print(f"Não foi possível calcular AUC para scoring: {e}")
    
    # Predições para conceding
    concede_predictions = model_concede.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva
    try:
        concede_auc = roc_auc_score(y_concede_test, concede_predictions)
        print(f"AUC Score (Conceding): {concede_auc:.4f}")
    except ValueError as e:
        print(f"Não foi possível calcular AUC para conceding: {e}")
    
    # Relatórios de classificação
    print("\nRelatório de Classificação - Scoring:")
    print(classification_report(y_score_test, model_score.predict(X_test)))
    
    print("\nRelatório de Classificação - Conceding:")
    print(classification_report(y_concede_test, model_concede.predict(X_test)))
    
    return model_score, model_concede

def save_model(models, model_path="models/"):
    """
    Salva os modelos treinados
    
    Args:
        models: Tupla com (model_score, model_concede)
        model_path (str): Caminho para salvar o modelo
    """
    os.makedirs(model_path, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_score, model_concede = models
    
    # Salvar modelo de scoring
    score_filename = f"vaep_scoring_model_{timestamp}.pkl"
    score_filepath = os.path.join(model_path, score_filename)
    with open(score_filepath, 'wb') as f:
        pickle.dump(model_score, f)
    
    # Salvar modelo de conceding
    concede_filename = f"vaep_conceding_model_{timestamp}.pkl"
    concede_filepath = os.path.join(model_path, concede_filename)
    with open(concede_filepath, 'wb') as f:
        pickle.dump(model_concede, f)
    
    print(f"Modelo de scoring salvo em: {score_filepath}")
    print(f"Modelo de conceding salvo em: {concede_filepath}")
    
    return score_filepath, concede_filepath

def calculate_vaep_values(models, actions):
    """
    Calcula valores VAEP para as ações
    
    Args:
        models: Tupla com (model_score, model_concede)
        actions (pd.DataFrame): DataFrame com dados SPADL
    
    Returns:
        pd.DataFrame: DataFrame com valores VAEP adicionados
    """
    print("Calculando valores VAEP...")
    
    model_score, model_concede = models
    home_team_id = actions.iloc[0]["home_team_id"]
    
    # Converter para game states
    gamestates = fs.gamestates(actions, nb_prev_actions=3)
    gamestates = fs.play_left_to_right(gamestates, home_team_id)
    
    # Extrair features

    xfns = [
        fs.actiontype_onehot,
        fs.bodypart_onehot,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        fs.time_delta
    ]
    X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
    
    # Prever probabilidades
    scoring_probs = model_score.predict_proba(X)[:, 1]  # Probabilidade da classe positiva
    conceding_probs = model_concede.predict_proba(X)[:, 1]  # Probabilidade da classe positiva
    
    # Calcular valores VAEP usando a fórmula básica
    # Valor ofensivo = diferença na probabilidade de marcar
    # Valor defensivo = diferença na probabilidade de sofrer (invertido)
    offensive_values = scoring_probs
    defensive_values = -conceding_probs  # Negativo porque menos probabilidade de sofrer é melhor
    vaep_values = offensive_values + defensive_values
    
    # Adicionar valores ao DataFrame original
    actions_with_vaep = actions.copy()
    
    # Alinhar com as ações originais (gamestates pode ter menos linhas)
    min_length = min(len(actions), len(vaep_values))
    
    # Inicializar com NaN
    actions_with_vaep['vaep_value'] = np.nan
    actions_with_vaep['offensive_value'] = np.nan
    actions_with_vaep['defensive_value'] = np.nan
    actions_with_vaep['scoring_prob'] = np.nan
    actions_with_vaep['conceding_prob'] = np.nan
    
    # Preencher valores onde possível
    if len(vaep_values) <= len(actions):
        start_idx = len(actions) - len(vaep_values)
        actions_with_vaep.iloc[start_idx:, actions_with_vaep.columns.get_loc('vaep_value')] = vaep_values
        actions_with_vaep.iloc[start_idx:, actions_with_vaep.columns.get_loc('offensive_value')] = offensive_values
        actions_with_vaep.iloc[start_idx:, actions_with_vaep.columns.get_loc('defensive_value')] = defensive_values
        actions_with_vaep.iloc[start_idx:, actions_with_vaep.columns.get_loc('scoring_prob')] = scoring_probs
        actions_with_vaep.iloc[start_idx:, actions_with_vaep.columns.get_loc('conceding_prob')] = conceding_probs
    
    valid_values = actions_with_vaep['vaep_value'].notna()
    print(f"Valores VAEP calculados para {valid_values.sum()} de {len(actions_with_vaep)} ações")
    
    if valid_values.sum() > 0:
        print(f"Valor VAEP médio: {actions_with_vaep['vaep_value'][valid_values].mean():.6f}")
        print(f"Valor VAEP máximo: {actions_with_vaep['vaep_value'][valid_values].max():.6f}")
        print(f"Valor VAEP mínimo: {actions_with_vaep['vaep_value'][valid_values].min():.6f}")
    
    return actions_with_vaep

# spadl_df = pd.read_parquet('./data/events_spadl.parquet')
# X, Y  = prepare_vaep_data(spadl_df)
# models = train_vaep_model(X, Y)
# model_paths = save_model(models)